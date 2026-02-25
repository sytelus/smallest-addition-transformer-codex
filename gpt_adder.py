import math
import random
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Minimal GPT-style decoder-only adder (<50 params, no checkpoint).

Key points:
- Model remains tiny (46 trainable parameters).
- No conventional training loop over large datasets/checkpoints.
- We solve weights using a tiny optimization over the 16 full-adder transition
  states in `compute_weights()`.
- Inference is standard autoregressive generation (`argmax` over next token).
"""


# ==========================================
# 1. TOKENIZER (binary pair encoding)
# ==========================================
class AdderTokenizer:
    """Encode A+B prompt into 35 bit-pair tokens + one start-state token."""

    prompt_len = 36
    gen_len = 35

    @staticmethod
    def parse_prompt(prompt: str) -> Tuple[int, int]:
        if not prompt.endswith("=") or "+" not in prompt:
            raise ValueError(f"Invalid prompt format: {prompt!r}")
        a_str, b_str = prompt[:-1].split("+")
        a = int(a_str)
        b = int(b_str)
        if not (0 <= a <= 9_999_999_999 and 0 <= b <= 9_999_999_999):
            raise ValueError("Operands must be in [0, 9_999_999_999]")
        return a, b

    def encode(self, strings: Sequence[str]) -> torch.Tensor:
        batch = []
        for s in strings:
            a, b = self.parse_prompt(s)
            a_bin = bin(a)[2:].zfill(35)[::-1]  # LSB first
            b_bin = bin(b)[2:].zfill(35)[::-1]  # LSB first

            # Token in {0,1,2,3} encodes one bit pair (a_i, b_i): 2*a_i + b_i
            seq = [int(x) * 2 + int(y) for x, y in zip(a_bin, b_bin)]

            # Initial state token O0 = 0 (sum_bit=0, carry=0).
            seq.append(0)
            batch.append(seq)

        return torch.tensor(batch, dtype=torch.long)

    @staticmethod
    def decode(token_ids: torch.Tensor) -> List[str]:
        """Decode generated state tokens into 11-digit decimal strings."""
        answers = []
        for seq in token_ids:
            gen_tokens = seq[36:]  # generated 35 state tokens
            bits = [str(int(t.item()) % 2) for t in gen_tokens]  # sum bits
            val = int("".join(bits[::-1]), 2)  # back to MSB-first
            answers.append(f"{val:011d}")
        return answers


# ==========================================
# 2. GPT-STYLE DECODER-ONLY MODEL (46 params)
# ==========================================
class GPTAdder(nn.Module):
    def __init__(self):
        super().__init__()
        # 8 params
        self.wte = nn.Embedding(4, 2)
        # 16 params (bias=False keeps it minimal)
        self.attn = nn.MultiheadAttention(embed_dim=2, num_heads=1, bias=False, batch_first=True)
        # 22 params
        self.mlp = nn.Sequential(
            nn.Linear(2, 4, bias=True),
            nn.ReLU(),
            nn.Linear(4, 2, bias=True),
        )
        # tied head (0 extra params)
        self.lm_head = nn.Linear(2, 4, bias=False)
        self.lm_head.weight = self.wte.weight

    @staticmethod
    def build_transition_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Sparse causal routing mask.

        For each position t, allow attention only to:
        - itself (state memory)
        - paired input position t-35 (bit-pair token)
        This yields the recurrence O_{i+1} = f(T_i, O_i).

        Note: True means masked in PyTorch MHA bool masks.
        """
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        i = torch.arange(seq_len, device=device)
        mask[i, i] = False
        src = i - 35
        valid = src >= 0
        mask[i[valid], src[valid]] = False
        return mask

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.wte(input_ids)
        seq_len = x.size(1)
        attn_mask = self.build_transition_mask(seq_len, x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(x)
        return self.lm_head(x)


# ==========================================
# 3. WEIGHT SOLVER (tiny optimization, no ckpt)
# ==========================================
def build_transition_supervision() -> Tuple[torch.Tensor, torch.Tensor]:
    """16 full-adder transitions (T,O)->Y as tiny supervised set."""
    contexts = []
    targets = []
    for a in [0, 1]:
        for b in [0, 1]:
            for s in [0, 1]:
                for c in [0, 1]:
                    t = a * 2 + b         # pair token
                    o = s + 2 * c         # current state token
                    y = (a + b + c) % 2 + 2 * ((a + b + c) // 2)  # next state

                    x = torch.zeros(36, dtype=torch.long)
                    x[0] = t
                    x[35] = o
                    contexts.append(x)
                    targets.append(y)
    return torch.stack(contexts), torch.tensor(targets, dtype=torch.long)


def transition_table_accuracy(model: GPTAdder, contexts: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(contexts)[:, -1, :]
        pred = logits.argmax(dim=-1)
        return float((pred == targets).float().mean().item())


def compute_weights(model: GPTAdder, max_restarts: int = 8, max_steps: int = 3000, lr: float = 5e-3) -> None:
    """Solve the tiny model by optimization on 16 transitions.

    This is not conventional training on large datasets. It is a direct
    parameter solve over the exact full-adder truth table.
    """
    contexts, targets = build_transition_supervision()
    best_acc = -1.0
    best_state = None

    for seed in range(1, max_restarts + 1):
        torch.manual_seed(seed)
        fresh = GPTAdder()
        model.load_state_dict(fresh.state_dict())

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(max_steps):
            logits = model(contexts)[:, -1, :]
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if bool((logits.argmax(dim=-1) == targets).all()):
                    print(f"Solved transition table exactly with seed={seed}")
                    return

        acc = transition_table_accuracy(model, contexts, targets)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    raise RuntimeError(f"Could not solve transitions exactly. Best transition accuracy: {best_acc:.4f}")


# ==========================================
# 4. AUTOREGRESSIVE GENERATION
# ==========================================
def generate(model: GPTAdder, tokenizer: AdderTokenizer, strings: Sequence[str]) -> List[str]:
    input_ids = tokenizer.encode(strings)
    model.eval()
    with torch.no_grad():
        for _ in range(tokenizer.gen_len):
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids)


# ==========================================
# 5. DEBUG / EVAL
# ==========================================
def expected_states(a: int, b: int) -> List[int]:
    out = []
    carry = 0
    for i in range(35):
        abit = (a >> i) & 1
        bbit = (b >> i) & 1
        s = abit + bbit + carry
        sum_bit = s & 1
        carry = s >> 1
        out.append(sum_bit + 2 * carry)
    return out


def debug_one(model: GPTAdder, tokenizer: AdderTokenizer, prompt: str) -> None:
    a, b = tokenizer.parse_prompt(prompt)
    expected = f"{a + b:011d}"
    ids = tokenizer.encode([prompt])
    trace = []
    model.eval()
    with torch.no_grad():
        for _ in range(tokenizer.gen_len):
            logits = model(ids)
            nxt = int(logits[0, -1, :].argmax().item())
            trace.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
    pred = tokenizer.decode(ids)[0]
    exp_trace = expected_states(a, b)
    print("Failure debug:")
    print(f"prompt={prompt}")
    print(f"pred={pred} expected={expected}")
    print(f"generated_states_first12={trace[:12]}")
    print(f"expected_states_first12={exp_trace[:12]}")
    print(f"generated_sum_bits_first12={[t % 2 for t in trace[:12]]}")
    print(f"expected_sum_bits_first12={[t % 2 for t in exp_trace[:12]]}")


def make_prompts(n: int, seed: int = 42) -> List[str]:
    random.seed(seed)
    out = []
    for _ in range(n):
        a = random.randint(0, 9_999_999_999)
        b = random.randint(0, 9_999_999_999)
        out.append(f"{a:010d}+{b:010d}=")
    return out


def run_stage(model: GPTAdder, tokenizer: AdderTokenizer, prompts: Sequence[str], n: int) -> bool:
    subset = list(prompts[:n])
    pred = generate(model, tokenizer, subset)
    exp = [f"{tokenizer.parse_prompt(p)[0] + tokenizer.parse_prompt(p)[1]:011d}" for p in subset]
    correct = sum(int(p == e) for p, e in zip(pred, exp))

    print("====================================")
    print(f"Stage {n}: {correct}/{n} correct")
    print("====================================")
    for i in range(min(n, 3)):
        print(f"Prompt : {subset[i]}")
        print(f"Output : {pred[i]}")
        print(f"Math   : {exp[i]}\n")

    if correct != n:
        first_bad = next(i for i, (p, e) in enumerate(zip(pred, exp)) if p != e)
        debug_one(model, tokenizer, subset[first_bad])
        return False
    return True


# ==========================================
# 6. MAIN
# ==========================================
if __name__ == "__main__":
    model = GPTAdder()
    compute_weights(model)
    tokenizer = AdderTokenizer()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total Standard Parameter Count: {param_count}\n")

    prompts = make_prompts(100, seed=42)

    # Requested progression
    if not run_stage(model, tokenizer, prompts, 1):
        raise SystemExit(1)
    if not run_stage(model, tokenizer, prompts, 2):
        raise SystemExit(1)
    if not run_stage(model, tokenizer, prompts, 3):
        raise SystemExit(1)

    # Extended checks
    ok10 = run_stage(model, tokenizer, prompts, 10)
    ok100 = run_stage(model, tokenizer, prompts, 100)
    if ok10 and ok100:
        print("All stages passed: 1, 2, 3, 10, 100.")
