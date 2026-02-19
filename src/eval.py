"""Evaluation and inference entrypoint for smallest addition transformer.

Usage:
  python -m src.eval test --ckpt checkpoints/best.pt
  python -m src.eval predict --ckpt checkpoints/best.pt --a 123 --b 45
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.data import (
    MAX_OPERAND,
    SUM_DIGITS,
    TARGET_LEN,
    POW10_11,
    ITOS,
    build_holdout_splits,
    postprocess,
    preprocess,
)
from src.model import ModelConfig, TinyDecoderLM


def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> TinyDecoderLM:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**blob["model_config"])
    model = TinyDecoderLM(mcfg).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_exact_match(
    model: TinyDecoderLM,
    a: torch.Tensor,
    b: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    n = int(a.numel())
    exact = 0
    token_correct = 0
    token_total = n * SUM_DIGITS

    from src.data import preprocess_batch  # local import to avoid cluttering namespace

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        aa = a[start:end]
        bb = b[start:end]

        prompt_t = preprocess_batch(aa, bb).to(device)
        gen = model.generate(prompt_t, max_new_tokens=TARGET_LEN)
        pred_digits = gen[:, -TARGET_LEN:-1].to("cpu")

        tgt_digits = ((aa[:, None] + bb[:, None]) // POW10_11[None, :]) % 10
        tgt_digits = tgt_digits.to(torch.long)

        matches = pred_digits.eq(tgt_digits)
        token_correct += int(matches.sum().item())
        exact += int(matches.all(dim=1).sum().item())

    return exact / n, token_correct / token_total


@torch.no_grad()
def per_digit_accuracy(
    model: TinyDecoderLM,
    a: torch.Tensor,
    b: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> List[float]:
    model.eval()
    n = int(a.numel())
    correct = torch.zeros(SUM_DIGITS, dtype=torch.long)
    total = torch.zeros(SUM_DIGITS, dtype=torch.long)

    from src.data import preprocess_batch

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        aa = a[start:end]
        bb = b[start:end]

        prompt_t = preprocess_batch(aa, bb).to(device)
        gen = model.generate(prompt_t, max_new_tokens=TARGET_LEN)
        pred_digits = gen[:, -TARGET_LEN:-1].to("cpu")

        tgt = ((aa[:, None] + bb[:, None]) // POW10_11[None, :]) % 10
        tgt = tgt.to(torch.long)
        m = pred_digits.eq(tgt)
        correct += m.sum(dim=0)
        total += torch.tensor([m.shape[0]] * SUM_DIGITS, dtype=torch.long)

    return [(int(c) / int(t)) for c, t in zip(correct.tolist(), total.tolist())]


@torch.no_grad()
def collect_failures(
    model: TinyDecoderLM,
    a: torch.Tensor,
    b: torch.Tensor,
    batch_size: int,
    device: torch.device,
    limit: int = 10,
) -> List[Dict[str, str]]:
    model.eval()
    fails: List[Dict[str, str]] = []
    n = int(a.numel())

    from src.data import preprocess_batch

    for start in range(0, n, batch_size):
        if len(fails) >= limit:
            break

        end = min(start + batch_size, n)
        aa = a[start:end]
        bb = b[start:end]

        prompt_t = preprocess_batch(aa, bb).to(device)
        gen = model.generate(prompt_t, max_new_tokens=TARGET_LEN)
        pred_tail = gen[:, -TARGET_LEN:].to("cpu")
        pred_digits = pred_tail[:, :SUM_DIGITS]

        tgt_digits = ((aa[:, None] + bb[:, None]) // POW10_11[None, :]) % 10
        tgt_digits = tgt_digits.to(torch.long)

        mismatch = ~pred_digits.eq(tgt_digits).all(dim=1)
        bad_idx = torch.nonzero(mismatch, as_tuple=False).flatten().tolist()

        for bi in bad_idx:
            ai = int(aa[bi].item())
            bj = int(bb[bi].item())
            pred_num = postprocess(pred_tail[bi].tolist())
            pstr = "".join(str(int(x)) if 0 <= int(x) <= 9 else ITOS[int(x)] for x in pred_digits[bi].tolist())
            tstr = "".join(str(int(x)) for x in tgt_digits[bi].tolist())
            fails.append(
                {
                    "A": str(ai),
                    "B": str(bj),
                    "prediction": str(pred_num),
                    "ground_truth": str(ai + bj),
                    "pred_rev": pstr,
                    "true_rev": tstr,
                }
            )
            if len(fails) >= limit:
                break

    return fails


def run_test(
    ckpt_path: Path,
    split_dir: Path,
    seed: int,
    val_size: int,
    test_size: int,
    eval_batch: int,
    device: str,
    out_json: Path,
) -> Dict:
    dev = torch.device(device)
    splits_path = split_dir / f"holdout_v{val_size}_t{test_size}_seed{seed}.pt"
    splits = build_holdout_splits(val_size, test_size, seed, splits_path)

    test_a = splits["test_a"]
    test_b = splits["test_b"]

    model = load_model_from_ckpt(ckpt_path, dev)
    em, tok_acc = evaluate_exact_match(model, test_a, test_b, eval_batch, dev)

    results = {
        "checkpoint": str(ckpt_path),
        "test_size": int(test_a.numel()),
        "exact_match": em,
        "token_accuracy": tok_acc,
    }

    if em < 0.99:
        results["per_digit_accuracy_lsd_to_msd"] = per_digit_accuracy(model, test_a, test_b, eval_batch, dev)

    results["failure_samples"] = collect_failures(model, test_a, test_b, eval_batch, dev, limit=10)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def predict_single(ckpt_path: Path, a: int, b: int, device: str = "cpu") -> Dict[str, int]:
    if not (0 <= a < MAX_OPERAND and 0 <= b < MAX_OPERAND):
        raise ValueError(f"A and B must be in [0, {MAX_OPERAND - 1}]")

    dev = torch.device(device)
    model = load_model_from_ckpt(ckpt_path, dev)

    prompt = torch.tensor([preprocess(a, b)], dtype=torch.long, device=dev)
    gen = model.generate(prompt, max_new_tokens=TARGET_LEN)
    tail = gen[0, -TARGET_LEN:].tolist()

    pred = postprocess(tail)
    return {"A": a, "B": b, "prediction": pred, "ground_truth": a + b, "correct": int(pred == a + b)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate smallest addition transformer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("test", help="Evaluate on held-out test set")
    pt.add_argument("--ckpt", type=Path, required=True)
    pt.add_argument("--split-dir", type=Path, default=Path("results/data"))
    pt.add_argument("--seed", type=int, default=123)
    pt.add_argument("--val-size", type=int, default=2000)
    pt.add_argument("--test-size", type=int, default=10000)
    pt.add_argument("--eval-batch-size", type=int, default=512)
    pt.add_argument("--device", type=str, default="cpu")
    pt.add_argument("--out-json", type=Path, default=Path("results/final_results.json"))

    pp = sub.add_parser("predict", help="Run inference on one (A,B) pair")
    pp.add_argument("--ckpt", type=Path, required=True)
    pp.add_argument("--a", type=int, required=True)
    pp.add_argument("--b", type=int, required=True)
    pp.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.cmd == "test":
        result = run_test(
            ckpt_path=args.ckpt,
            split_dir=args.split_dir,
            seed=args.seed,
            val_size=args.val_size,
            test_size=args.test_size,
            eval_batch=args.eval_batch_size,
            device=args.device,
            out_json=args.out_json,
        )
        print(json.dumps(result, indent=2))

    elif args.cmd == "predict":
        result = predict_single(args.ckpt, args.a, args.b, args.device)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
