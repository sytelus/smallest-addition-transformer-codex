import math
import random
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Tokenizer (outside the model; allowed to reshape inputs)
# ============================================================
class AdderTokenizer:
    """
    External interface:
      - encode_prompts(["0000000123+0000000045=", ...]) -> LongTensor[B, 22]
      - decode_output_tokens(out_ids[B, 11]) -> List[str]  (11-digit, MSD-first)

    Internally we use a vocab that includes carry-encoded output tokens so the
    model doesn't need any special carry logic in forward().

    Vocab:
      0..9    : A-digit tokens A0..A9 (used only in prompt)
      10..19  : B-digit tokens B0..B9 (used only in prompt)
      20      : '+'
      21      : '='
      22..31  : output digit 0..9 with carry_out=0  (O0c0..O9c0)
      32..41  : output digit 0..9 with carry_out=1  (O0c1..O9c1)
    """
    def __init__(self):
        self.vocab_size = 42

    def encode_prompt(self, s: str) -> List[int]:
        # Expect exactly: 10 digits + '+' + 10 digits + '='  => len = 22
        if len(s) != 22 or s[10] != "+" or s[-1] != "=":
            raise ValueError(f"Bad prompt format: {s!r}")

        a = s[:10]
        b = s[11:21]

        # Reverse digits so column k is at fixed relative offset in attention wiring
        a_rev = list(reversed(a))
        b_rev = list(reversed(b))

        ids = [int(ch) for ch in a_rev] + [20] + [10 + int(ch) for ch in b_rev] + [21]
        return ids  # length 22

    def encode_prompts(self, prompts: List[str]) -> torch.LongTensor:
        return torch.tensor([self.encode_prompt(p) for p in prompts], dtype=torch.long)

    def decode_output_tokens(self, out_ids: torch.LongTensor) -> List[str]:
        """
        out_ids: (B, 11) tokens from the model, generated after the '='.
        These tokens represent digits LSD-first; we reverse to MSD-first.
        """
        out = []
        for row in out_ids.tolist():
            lsd_digits = []
            for t in row:
                if 22 <= t <= 31:
                    lsd_digits.append(str(t - 22))
                elif 32 <= t <= 41:
                    lsd_digits.append(str(t - 32))
                else:
                    lsd_digits.append("?")
            out.append("".join(reversed(lsd_digits)))
        return out


# ============================================================
# Generic GPT-style model (NO addition-specific code here)
# ============================================================
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    n_inner: int
    dropout: float = 0.0
    use_bias: bool = False
    activation: str = "relu"   # "relu" or "gelu"
    tie_weights: bool = False  # optional


class CausalSelfAttention(nn.Module):
    """
    Standard causal multi-head self-attention.
    Has a standard additive attention mask buffer (like many GPT implementations).
    By default it's just the usual causal mask.
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.block_size = cfg.block_size

        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.use_bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.use_bias)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.use_bias)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.use_bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Standard causal attention mask: allowed => 0, blocked => -1e9
        tril = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        attn_mask = (1.0 - tril) * -1e9
        self.register_buffer(
            "attn_mask",
            attn_mask.view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        nh, hd = self.n_head, self.head_dim

        q = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)  # (B, nh, T, hd)
        k = self.k_proj(x).view(B, T, nh, hd).transpose(1, 2)  # (B, nh, T, hd)
        v = self.v_proj(x).view(B, T, nh, hd).transpose(1, 2)  # (B, nh, T, hd)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)         # (B, nh, T, T)
        att = att + self.attn_mask[:, :, :T, :T]
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v                                             # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)         # (B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, cfg.n_inner, bias=True)
        self.fc2 = nn.Linear(cfg.n_inner, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.activation == "gelu":
            self.act = F.gelu
        else:
            self.act = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class GPT(nn.Module):
    """
    Standard GPT-style decoder-only Transformer:
      token embedding -> blocks -> lm_head

    forward() contains no problem-specific logic.
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        x = self.tok_emb(idx)               # (B, T, C)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(x)            # (B, T, vocab)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ============================================================
# Weight programming (addition-specific; OUTSIDE the model)
# ============================================================
def program_weights_for_10digit_addition(model: GPT, tokenizer: AdderTokenizer) -> None:
    """
    Programs a generic GPT instance to do exact 10-digit addition on the
    tokenizer's 42-token vocabulary.

    This function is the only place that "knows" anything about addition.
    """
    V = tokenizer.vocab_size
    feature_dim = 22  # [a_onehot(10), b_onehot(10), carry0, carry1]
    D = model.cfg.n_embd
    bs = model.cfg.block_size

    if V != 42:
        raise ValueError("This programming expects vocab_size=42 from AdderTokenizer.")
    if D != feature_dim + V:
        raise ValueError(f"This programming expects n_embd={feature_dim + V}.")
    if model.cfg.n_layer != 2 or model.cfg.n_head != 1 or model.cfg.n_inner < 202:
        raise ValueError("This programming expects n_layer=2, n_head=1, n_inner>=202.")

    with torch.no_grad():
        # Zero all trainable parameters
        for p in model.parameters():
            p.zero_()

        # -------------------------
        # Token embedding table
        # -------------------------
        W = model.tok_emb.weight  # (V, D)

        # A digits 0..9: a_onehot
        for d in range(10):
            W[d, d] = 1.0

        # B digits 10..19: b_onehot
        for d in range(10):
            W[10 + d, 10 + d] = 1.0

        # '+' (20): all zeros

        # '=' (21): carry0=0.5 so a self-copy+residual doubles it to 1.0
        W[21, 20] = 0.5

        # Output tokens carry encoded:
        # 22..31: digit d with carry_out=0  => carry0=1
        for d in range(10):
            W[22 + d, 20] = 1.0
        # 32..41: digit d with carry_out=1  => carry1=1
        for d in range(10):
            W[32 + d, 21] = 1.0

        # -------------------------
        # lm_head: read logits from "logit subspace"
        # logits[t] = x[feature_dim + t]
        # -------------------------
        for t in range(V):
            model.lm_head.weight[t, feature_dim + t] = 1.0

        # Helper: turn an attention layer into pure "copy from allowed positions"
        def set_copy_attention(attn: CausalSelfAttention, v_mask: torch.Tensor = None):
            # Make scores independent of content: Q=0, K=0
            attn.q_proj.weight.zero_()
            attn.k_proj.weight.zero_()
            if attn.q_proj.bias is not None:
                attn.q_proj.bias.zero_()
            if attn.k_proj.bias is not None:
                attn.k_proj.bias.zero_()

            # V projection: identity or masked identity
            if v_mask is None:
                attn.v_proj.weight.copy_(torch.eye(D))
            else:
                attn.v_proj.weight.copy_(v_mask)
            if attn.v_proj.bias is not None:
                attn.v_proj.bias.zero_()

            # Output projection: identity
            attn.out_proj.weight.copy_(torch.eye(D))
            if attn.out_proj.bias is not None:
                attn.out_proj.bias.zero_()

        # -------------------------
        # Block 0 attention mask:
        # For i=11..20 (B digits), copy from i-11 (matching A digit).
        # For i=21 ('='), copy from itself (to double carry0 from 0.5 -> 1.0).
        # For i>21 (generated positions), copy from '+' position (10) which is all zeros.
        # Else copy from itself.
        # -------------------------
        mask0 = torch.full((bs, bs), -1e9)
        for i in range(bs):
            if 11 <= i <= 20:
                j = i - 11
            elif i == 21:
                j = 21
            elif i > 21:
                j = 10
            else:
                j = i
            if 0 <= j <= i:
                mask0[i, j] = 0.0
            else:
                mask0[i, i] = 0.0
        model.blocks[0].attn.attn_mask.copy_(mask0.view(1, 1, bs, bs))
        set_copy_attention(model.blocks[0].attn, v_mask=None)

        # Block 0 MLP unused: zero it
        model.blocks[0].mlp.fc1.weight.zero_()
        model.blocks[0].mlp.fc1.bias.zero_()
        model.blocks[0].mlp.fc2.weight.zero_()

        # -------------------------
        # Block 1 attention mask:
        # For i>=21, copy from i-10.
        # That maps:
        #   i=21 ('=')   -> j=11 (B digit column 0)
        #   i=22 (out0)  -> j=12 (B digit column 1)
        #   ...
        #   i=30 (out8)  -> j=20 (B digit column 9)
        #   i=31 (out9)  -> j=21 ('=')  (forces "no digit" path used for final carry)
        # For i<21, self-copy.
        # -------------------------
        mask1 = torch.full((bs, bs), -1e9)
        for i in range(bs):
            j = (i - 10) if i >= 21 else i
            if 0 <= j <= i:
                mask1[i, j] = 0.0
            else:
                mask1[i, i] = 0.0
        model.blocks[1].attn.attn_mask.copy_(mask1.view(1, 1, bs, bs))

        # V mask: do NOT copy carry bits (20,21) or logit subspace (feature_dim:)
        vmask = torch.eye(D)
        vmask[feature_dim:, :] = 0.0
        vmask[20, :] = 0.0
        vmask[21, :] = 0.0
        set_copy_attention(model.blocks[1].attn, v_mask=vmask)

        # -------------------------
        # Block 1 MLP: hard-coded lookup table:
        # Input features at the prediction position contain:
        #   - digit onehots for a,b (copied in by attention)
        #   - carry_in bit (from the current token embedding, i.e. previous output)
        #
        # We create 200 hidden units for all (a,b,carry_in) combos, plus 2 units for final carry.
        # Each active hidden unit outputs +1 logit into the correct output token slot.
        # -------------------------
        fc1 = model.blocks[1].mlp.fc1  # (n_inner, D)
        fc2 = model.blocks[1].mlp.fc2  # (D, n_inner)

        # Units 0..199: (carry_in, a, b)
        for c in (0, 1):
            for a in range(10):
                for b in range(10):
                    h = c * 100 + a * 10 + b
                    fc1.weight[h, a] = 1.0
                    fc1.weight[h, 10 + b] = 1.0
                    fc1.weight[h, 20 + c] = 1.0
                    fc1.bias[h] = -2.5  # so only exact combo fires with ReLU value 0.5

                    s = a + b + c
                    digit = s % 10
                    carry_out = 1 if s >= 10 else 0
                    out_id = 22 + carry_out * 10 + digit
                    logit_dim = feature_dim + out_id
                    fc2.weight[logit_dim, h] = 2.0  # 0.5 -> 1.0 logit

        # Unit 200: final step when no digits are present and carry_in=0 -> output O0c0 (id 22)
        h0 = 200
        fc1.weight[h0, 0:20] = -1.0
        fc1.weight[h0, 20] = 1.0
        fc1.bias[h0] = -0.5
        fc2.weight[feature_dim + 22, h0] = 2.0

        # Unit 201: final step when no digits are present and carry_in=1 -> output O1c0 (id 23)
        h1 = 201
        fc1.weight[h1, 0:20] = -1.0
        fc1.weight[h1, 21] = 1.0
        fc1.bias[h1] = -0.5
        fc2.weight[feature_dim + 23, h1] = 2.0

        # Any extra hidden units (if n_inner > 202) are disabled
        if fc1.weight.shape[0] > 202:
            fc1.bias[202:] = -1e9


# ============================================================
# Standard autoregressive generate() (outside the model)
# ============================================================
@torch.no_grad()
def generate(model: GPT, idx: torch.LongTensor, max_new_tokens: int) -> torch.LongTensor:
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx)                          # (B, T, vocab)
        next_id = logits[:, -1, :].argmax(dim=-1)    # greedy
        idx = torch.cat([idx, next_id[:, None]], dim=1)
    return idx


# ============================================================
# Demo + accuracy test
# ============================================================
def make_prompt(a: int, b: int) -> str:
    return f"{a:010d}+{b:010d}="


def main():
    tok = AdderTokenizer()

    # We choose dimensions to make the programming simple and exact.
    # n_embd = feature_dim(22) + vocab_size(42) = 64
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=64,   # >= 22(prompt) + 11(output) = 33
        n_layer=2,
        n_head=1,
        n_embd=64,
        n_inner=202,
        dropout=0.0,
        use_bias=False,
        activation="relu",
        tie_weights=False,
    )
    model = GPT(cfg)

    # Program the weights (outside the model definition)
    program_weights_for_10digit_addition(model, tok)

    # Print parameter count
    print("Trainable parameter count:", count_parameters(model))

    # Small demo
    prompts = [
        "0000000123+0000000045=",
        "9999999999+9999999999=",
        "0000000000+0000000000=",
        "0000000001+0000000009=",
    ]
    idx = tok.encode_prompts(prompts)
    out = generate(model, idx, max_new_tokens=11)
    out_ids = out[:, 22:33]
    answers = tok.decode_output_tokens(out_ids)
    for p, a in zip(prompts, answers):
        print(p, "->", a)

    # Accuracy test (vectorized, compares integers)
    N = 5000
    B = 250
    correct = 0
    total = 0

    for _ in range((N + B - 1) // B):
        bsz = min(B, N - total)
        if bsz <= 0:
            break

        a = torch.randint(0, 10**10, (bsz,), dtype=torch.long)
        b = torch.randint(0, 10**10, (bsz,), dtype=torch.long)

        prompts_batch = [make_prompt(int(ai), int(bi)) for ai, bi in zip(a.tolist(), b.tolist())]
        idx = tok.encode_prompts(prompts_batch)

        out = generate(model, idx, max_new_tokens=11)
        out_ids = out[:, 22:33]  # (bsz, 11)

        # Convert tokens -> digits LSD-first (strip carry bit)
        digits_lsd = torch.where(out_ids >= 32, out_ids - 32, out_ids - 22)  # 0..9
        digits_msd = torch.flip(digits_lsd, dims=[1]).long()

        # Convert 11 digits to integer
        powers = (10 ** torch.arange(10, -1, -1)).long()  # 10^10..10^0
        pred = (digits_msd * powers).sum(dim=1)

        truth = a + b
        correct += int((pred == truth).sum().item())
        total += bsz

    acc = correct / total
    print(f"Accuracy on {total} random cases: {acc:.6f}")


if __name__ == "__main__":
    main()