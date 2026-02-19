"""Data and tokenization pipeline for 10-digit addition.

This module defines deterministic preprocess/postprocess functions plus batch encoders.
"""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


MAX_OPERAND = 10_000_000_000  # 10^10
NUM_DIGITS = 10
SUM_DIGITS = 11

# Output tokens are decimal digits.
DIGIT_TOKENS = [str(i) for i in range(10)]

# Input tokens encode one digit-column pair (a_i, b_i), LSD -> MSD.
PAIR_TOKENS = [f"P{a}{b}" for a in range(10) for b in range(10)]

EQUALS = "="
BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"

VOCAB = DIGIT_TOKENS + PAIR_TOKENS + [EQUALS, BOS, EOS, PAD]
STOI: Dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
ITOS: Dict[int, str] = {i: tok for tok, i in STOI.items()}
VOCAB_SIZE = len(VOCAB)

PAIR_BASE = len(DIGIT_TOKENS)
EQUALS_ID = STOI[EQUALS]
BOS_ID = STOI[BOS]
EOS_ID = STOI[EOS]
PAD_ID = STOI[PAD]

PROMPT_LEN = 1 + NUM_DIGITS + 1  # <bos> + 10 pair tokens + '='
TARGET_LEN = SUM_DIGITS + 1      # 11 digits + <eos>
FULL_LEN = PROMPT_LEN + TARGET_LEN
INPUT_LEN = FULL_LEN - 1

POW10_10 = torch.tensor([10**i for i in range(NUM_DIGITS)], dtype=torch.int64)
POW10_11 = torch.tensor([10**i for i in range(SUM_DIGITS)], dtype=torch.int64)


def pair_token_id(a_digit: int, b_digit: int) -> int:
    return PAIR_BASE + (a_digit * 10 + b_digit)


def digits_rev_list(num: int, width: int) -> List[int]:
    return [int(ch) for ch in f"{num:0{width}d}"[::-1]]


def preprocess(a: int, b: int) -> List[int]:
    """Deterministic preprocess(A,B) -> prompt token IDs.

    Format: <bos> P(a0,b0) P(a1,b1) ... P(a9,b9) =
    where i=0 is least-significant digit column.
    """
    ad = digits_rev_list(a, NUM_DIGITS)
    bd = digits_rev_list(b, NUM_DIGITS)
    out = [BOS_ID]
    out.extend(pair_token_id(da, db) for da, db in zip(ad, bd))
    out.append(EQUALS_ID)
    return out


def preprocess_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized preprocess for batches of int64 operands on CPU."""
    ad = ((a[:, None] // POW10_10[None, :]) % 10).to(torch.long)
    bd = ((b[:, None] // POW10_10[None, :]) % 10).to(torch.long)
    pair_ids = PAIR_BASE + ad * 10 + bd

    bsz = a.shape[0]
    bos = torch.full((bsz, 1), BOS_ID, dtype=torch.long)
    eq = torch.full((bsz, 1), EQUALS_ID, dtype=torch.long)
    return torch.cat([bos, pair_ids, eq], dim=1)


def encode_batch(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build supervised LM tensors (x,y), with labels masked on prompt portion."""
    prompt = preprocess_batch(a, b)
    sums = a + b
    sum_digits = ((sums[:, None] // POW10_11[None, :]) % 10).to(torch.long)

    bsz = a.shape[0]
    eos = torch.full((bsz, 1), EOS_ID, dtype=torch.long)
    target = torch.cat([sum_digits, eos], dim=1)

    full = torch.cat([prompt, target], dim=1)
    x = full[:, :-1].clone()
    y = full[:, 1:].clone()
    y[:, : PROMPT_LEN - 1] = -100
    return x, y


def postprocess(generated: Sequence[int]) -> int:
    """Deterministic postprocess(model_output_tokens) -> integer C."""
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


def pair_hash(a: int, b: int) -> int:
    return a * MAX_OPERAND + b


def build_holdout_splits(val_size: int, test_size: int, seed: int, out_path: Path) -> Dict[str, torch.Tensor]:
    """Create/load deterministic holdout splits of (A,B) pairs."""
    if out_path.exists():
        data = torch.load(out_path, map_location="cpu", weights_only=False)
        if int(data["val_a"].numel()) == val_size and int(data["test_a"].numel()) == test_size:
            return data

    g = torch.Generator().manual_seed(seed)
    total = val_size + test_size

    pairs: List[Tuple[int, int]] = []
    seen = set()
    while len(pairs) < total:
        need = total - len(pairs)
        sample_n = max(need * 2, 4096)
        a = torch.randint(0, MAX_OPERAND, (sample_n,), generator=g, dtype=torch.int64)
        b = torch.randint(0, MAX_OPERAND, (sample_n,), generator=g, dtype=torch.int64)

        for ai, bi in zip(a.tolist(), b.tolist()):
            h = pair_hash(ai, bi)
            if h in seen:
                continue
            seen.add(h)
            pairs.append((ai, bi))
            if len(pairs) >= total:
                break

    val_pairs = pairs[:val_size]
    test_pairs = pairs[val_size:]

    data = {
        "val_a": torch.tensor([p[0] for p in val_pairs], dtype=torch.int64),
        "val_b": torch.tensor([p[1] for p in val_pairs], dtype=torch.int64),
        "test_a": torch.tensor([p[0] for p in test_pairs], dtype=torch.int64),
        "test_b": torch.tensor([p[1] for p in test_pairs], dtype=torch.int64),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    return data
