"""tiny_adder.py

Single-file, minimal, and fully runnable implementation of the tiny
decoder-only transformer used for 10-digit addition experiments.

Purpose
- Provide a compact reference that contains all core pieces in one place:
  model, tokenization, inference, and testing.
- Keep behavior aligned with the multi-file project code while reducing
  indirection for quick study and experimentation.

Goals
- Deterministic preprocessing/postprocessing for addition pairs (A, B).
- Tiny GPT-style decoder with causal self-attention and greedy generation.
- End-to-end CPU execution with a batch test harness.
- A `load_model()` placeholder where real checkpoint loading can be inserted.

Constraints
- Operands are non-negative integers in [0, 10^10 - 1] (fixed 10 digits).
- Sum is represented in 11 reversed digits (LSD -> MSD), then `<eos>`.
- Vocabulary is fixed:
  - output digits: 0..9
  - input pair tokens: P00..P99
  - specials: "=", "<bos>", "<eos>", "<pad>"
- This script defaults to random initialization in `load_model()`, so exact
  arithmetic accuracy is expected to be poor unless trained weights are loaded.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Task constants and vocabulary
# -----------------------------

MAX_OPERAND = 10_000_000_000  # 10^10
NUM_DIGITS = 10
SUM_DIGITS = 11

DIGIT_TOKENS = [str(i) for i in range(10)]
PAIR_TOKENS = [f"P{a}{b}" for a in range(10) for b in range(10)]
SPECIAL_TOKENS = ["=", "<bos>", "<eos>", "<pad>"]

VOCAB = DIGIT_TOKENS + PAIR_TOKENS + SPECIAL_TOKENS
STOI: Dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
ITOS: Dict[int, str] = {i: tok for tok, i in STOI.items()}
VOCAB_SIZE = len(VOCAB)  # 114

PAIR_BASE = len(DIGIT_TOKENS)  # pair token IDs start at 10
EQUALS_ID = STOI["="]
BOS_ID = STOI["<bos>"]
EOS_ID = STOI["<eos>"]
PAD_ID = STOI["<pad>"]

PROMPT_LEN = 1 + NUM_DIGITS + 1  # <bos> + 10 pair tokens + "="
TARGET_LEN = SUM_DIGITS + 1      # 11 digits + <eos>
FULL_LEN = PROMPT_LEN + TARGET_LEN
INPUT_LEN = FULL_LEN - 1         # autoregressive input length during training

POW10_10 = torch.tensor([10**i for i in range(NUM_DIGITS)], dtype=torch.int64)
POW10_11 = torch.tensor([10**i for i in range(SUM_DIGITS)], dtype=torch.int64)


# -----------------------------
# Data pipeline
# -----------------------------

def pair_token_id(a_digit: int, b_digit: int) -> int:
    """Map one digit-column pair to token ID."""
    return PAIR_BASE + (a_digit * 10 + b_digit)


def preprocess(a: int, b: int) -> List[int]:
    """Deterministic prompt encoding for one pair.

    Format:
    `<bos> P(a0,b0) P(a1,b1) ... P(a9,b9) =`
    where `a0/b0` are least-significant digits.
    """
    if not (0 <= a < MAX_OPERAND and 0 <= b < MAX_OPERAND):
        raise ValueError(f"A and B must be in [0, {MAX_OPERAND - 1}]")

    out = [BOS_ID]
    for i in range(NUM_DIGITS):
        da = (a // (10**i)) % 10
        db = (b // (10**i)) % 10
        out.append(pair_token_id(da, db))
    out.append(EQUALS_ID)
    return out


def preprocess_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized preprocess for int64 tensors `a` and `b` with shape [B]."""
    if a.dtype != torch.int64 or b.dtype != torch.int64:
        raise TypeError("preprocess_batch expects int64 tensors")
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1D tensors with same shape")

    ad = ((a[:, None] // POW10_10[None, :]) % 10).to(torch.long)
    bd = ((b[:, None] // POW10_10[None, :]) % 10).to(torch.long)
    pair_ids = PAIR_BASE + ad * 10 + bd

    bsz = a.shape[0]
    bos = torch.full((bsz, 1), BOS_ID, dtype=torch.long)
    eq = torch.full((bsz, 1), EQUALS_ID, dtype=torch.long)
    return torch.cat([bos, pair_ids, eq], dim=1)


def postprocess(generated: Sequence[int]) -> int:
    """Deterministic decode from generated token IDs to integer sum.

    Reads tokens until `<eos>` or first non-digit token. Uses up to 11 digits.
    If fewer than 11 digits are available, pads trailing zeros in reversed
    representation, then converts back to normal integer order.
    """
    digits: List[str] = []
    for tok in generated:
        tid = int(tok)
        if tid == EOS_ID:
            break
        if 0 <= tid <= 9:
            digits.append(str(tid))
        else:
            break

    if not digits:
        return 0

    if len(digits) < SUM_DIGITS:
        digits.extend(["0"] * (SUM_DIGITS - len(digits)))
    digits = digits[:SUM_DIGITS]
    return int("".join(digits)[::-1])


def decode_batch_tails(tails: torch.Tensor) -> torch.Tensor:
    """Decode a batch of generated tails `[B, TARGET_LEN]` to integer sums."""
    preds = [postprocess(row.tolist()) for row in tails]
    return torch.tensor(preds, dtype=torch.int64)


# -----------------------------
# Model
# -----------------------------

@dataclass
class ModelConfig:
    n_layer: int = 1
    d_model: int = 8
    n_head: int = 2
    d_ff: int = 12
    dropout: float = 0.0
    max_seq_len: int = INPUT_LEN
    vocab_size: int = VOCAB_SIZE


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask avoids attending to future tokens during autoregression.
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyAdderLM(nn.Module):
    """Tiny decoder-only LM for the addition tokenization above."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying reduces parameters and typically improves small LMs.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, seqlen = idx.shape

        if seqlen > self.cfg.max_seq_len:
            idx = idx[:, -self.cfg.max_seq_len :]
            if targets is not None:
                targets = targets[:, -self.cfg.max_seq_len :]
            seqlen = idx.shape[1]

        pos = torch.arange(seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy autoregressive decoding."""
        out = prompt
        for _ in range(max_new_tokens):
            idx = out[:, -self.cfg.max_seq_len :]
            logits, _ = self.forward(idx)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, next_tok], dim=1)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# -----------------------------
# Loading and inference
# -----------------------------

def load_model(config: Optional[ModelConfig] = None, device: str = "cpu", seed: int = 123) -> TinyAdderLM:
    """Placeholder model loader.

    Replace this with checkpoint loading when trained weights are available.
    For now it initializes trainable parameters randomly (but reproducibly).
    """
    if config is None:
        config = ModelConfig()
    torch.manual_seed(seed)
    model = TinyAdderLM(config).to(torch.device(device))
    model.eval()
    return model


@torch.no_grad()
def predict_sum(model: TinyAdderLM, a: int, b: int) -> Dict[str, int]:
    """Run inference for one pair and return decoded result."""
    if not (0 <= a < MAX_OPERAND and 0 <= b < MAX_OPERAND):
        raise ValueError(f"A and B must be in [0, {MAX_OPERAND - 1}]")

    device = next(model.parameters()).device
    prompt = torch.tensor([preprocess(a, b)], dtype=torch.long, device=device)
    generated = model.generate(prompt, max_new_tokens=TARGET_LEN)
    tail = generated[0, -TARGET_LEN:].to("cpu")
    pred = postprocess(tail.tolist())
    truth = a + b
    return {
        "A": int(a),
        "B": int(b),
        "prediction": int(pred),
        "ground_truth": int(truth),
        "correct": int(pred == truth),
    }


@torch.no_grad()
def batch_predict(model: TinyAdderLM, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return `(predictions, generated_tails)` for a batch of operand pairs."""
    device = next(model.parameters()).device
    prompt = preprocess_batch(a, b).to(device)
    generated = model.generate(prompt, max_new_tokens=TARGET_LEN)
    tails = generated[:, -TARGET_LEN:].to("cpu")
    preds = decode_batch_tails(tails)
    return preds, tails


# -----------------------------
# Test harness
# -----------------------------

@torch.no_grad()
def run_batch_test(model: TinyAdderLM, batch_size: int = 128, seed: int = 7) -> Dict[str, float]:
    """Generate a random batch, run inference, and verify pipeline outputs.

    Verification performed:
    - Output tensor shape and dtype checks.
    - Predictions are within valid sum range [0, 2*MAX_OPERAND-2].
    - Deterministic encode/decode sanity using oracle tokens:
      if true reversed digits + `<eos>` are decoded, we recover exact sums.

    Returns summary metrics including exact-match and token-level accuracy.
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, MAX_OPERAND, (batch_size,), generator=g, dtype=torch.int64)
    b = torch.randint(0, MAX_OPERAND, (batch_size,), generator=g, dtype=torch.int64)
    truth = a + b

    preds, tails = batch_predict(model, a, b)

    if preds.shape != truth.shape:
        raise AssertionError(f"Prediction shape mismatch: {preds.shape} vs {truth.shape}")
    if preds.dtype != torch.int64:
        raise AssertionError(f"Prediction dtype should be int64, got {preds.dtype}")
    if not torch.all((preds >= 0) & (preds <= (2 * MAX_OPERAND - 2))):
        raise AssertionError("Predictions out of valid integer-sum range")

    # Token accuracy is measured on 11 reversed sum digits (excluding <eos>).
    pred_digits = tails[:, :SUM_DIGITS]
    true_digits = ((truth[:, None] // POW10_11[None, :]) % 10).to(torch.long)
    token_accuracy = float(pred_digits.eq(true_digits).float().mean().item())

    # Oracle sanity check: decoding true targets must perfectly reconstruct sums.
    eos = torch.full((batch_size, 1), EOS_ID, dtype=torch.long)
    oracle_tails = torch.cat([true_digits, eos], dim=1)
    oracle_decoded = decode_batch_tails(oracle_tails)
    oracle_roundtrip_ok = bool(torch.equal(oracle_decoded, truth))
    if not oracle_roundtrip_ok:
        raise AssertionError("Oracle roundtrip failed: postprocess is inconsistent")

    exact_match = float(preds.eq(truth).float().mean().item())
    return {
        "batch_size": float(batch_size),
        "exact_match": exact_match,
        "token_accuracy": token_accuracy,
        "oracle_roundtrip_ok": float(oracle_roundtrip_ok),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-file tiny transformer adder demo")
    parser.add_argument("--batch-size", type=int, default=128, help="Random examples used in batch test")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for generated test examples")
    parser.add_argument("--model-seed", type=int, default=123, help="Random seed for model init in load_model()")
    parser.add_argument("--examples", type=int, default=5, help="Number of single examples to print")
    args = parser.parse_args()

    cfg = ModelConfig()
    model = load_model(config=cfg, device="cpu", seed=args.model_seed)

    print(f"Device: cpu")
    print(f"Parameters: {count_parameters(model)}")
    print("Running batch test...")
    summary = run_batch_test(model, batch_size=args.batch_size, seed=args.seed)
    print("Batch test summary:")
    print(json.dumps(summary, indent=2))

    print("\nSample predictions:")
    g = torch.Generator().manual_seed(args.seed + 999)
    for _ in range(args.examples):
        a = int(torch.randint(0, MAX_OPERAND, (1,), generator=g, dtype=torch.int64).item())
        b = int(torch.randint(0, MAX_OPERAND, (1,), generator=g, dtype=torch.int64).item())
        result = predict_sum(model, a, b)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
