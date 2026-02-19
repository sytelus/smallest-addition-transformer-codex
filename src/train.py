"""Training entrypoint for smallest addition transformer.

Usage example:
  python -m src.train --run-name repro --train-steps 5000
"""

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.data import (
    MAX_OPERAND,
    INPUT_LEN,
    VOCAB_SIZE,
    build_holdout_splits,
    encode_batch,
    pair_hash,
)
from src.eval import evaluate_exact_match
from src.model import ModelConfig, TinyDecoderLM, count_parameters


@dataclass
class TrainConfig:
    seed: int
    train_steps: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    min_lr_ratio: float
    grad_clip: float
    eval_interval: int
    val_size: int
    test_size: int
    eval_batch_size: int
    run_name: str
    run_dir: str
    split_dir: str
    best_ckpt_out: str
    last_ckpt_out: str
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class TrainBatchSampler:
    def __init__(self, batch_size: int, seed: int, reserved_hashes: set):
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.reserved_hashes = reserved_hashes

    def sample_pairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)
        b = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)

        # Strictly avoid holdout leakage.
        for i in range(self.batch_size):
            ai = int(a[i].item())
            bi = int(b[i].item())
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
            a[i] = ai
            b[i] = bi
        return a, b

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.sample_pairs()
        return encode_batch(a, b)


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv_header(path: Path, header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv(path: Path, row: List) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def train(model_cfg: ModelConfig, train_cfg: TrainConfig) -> Dict:
    device = torch.device(train_cfg.device)
    run_dir = Path(train_cfg.run_dir)
    split_dir = Path(train_cfg.split_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.csv"
    set_seed(train_cfg.seed)

    split_path = split_dir / f"holdout_v{train_cfg.val_size}_t{train_cfg.test_size}_seed{train_cfg.seed}.pt"
    splits = build_holdout_splits(train_cfg.val_size, train_cfg.test_size, train_cfg.seed, split_path)

    reserved_hashes = set()
    for ai, bi in zip(splits["val_a"].tolist(), splits["val_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))
    for ai, bi in zip(splits["test_a"].tolist(), splits["test_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))

    val_a, val_b = splits["val_a"], splits["val_b"]

    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
    save_csv_header(metrics_path, ["step", "train_loss", "val_exact", "val_token_acc", "lr", "elapsed_sec"])

    best_val = -1.0
    best_step = -1
    t0 = time.time()

    print(f"Run: {train_cfg.run_name}")
    print(f"Params: {params}")
    print(f"Device: {device}")

    for step in range(train_cfg.train_steps):
        model.train()
        x, y = sampler.sample_batch()
        x = x.to(device)
        y = y.to(device)

        lr_now = cosine_lr(step, train_cfg.train_steps, train_cfg.lr, train_cfg.warmup_steps, train_cfg.min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
            val_exact, val_tok = evaluate_exact_match(model, val_a, val_b, train_cfg.eval_batch_size, device)
            elapsed = time.time() - t0
            train_loss = float(loss.item())
            append_csv(metrics_path, [step, train_loss, val_exact, val_tok, lr_now, elapsed])
            print(
                f"step={step:6d} loss={train_loss:.4f} val_exact={val_exact:.5f} "
                f"val_tok={val_tok:.5f} lr={lr_now:.2e} t={elapsed:.1f}s"
            )

            if val_exact > best_val:
                best_val = val_exact
                best_step = step
                Path(train_cfg.best_ckpt_out).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": asdict(model_cfg),
                        "train_config": asdict(train_cfg),
                        "step": step,
                        "val_exact": val_exact,
                        "params": params,
                    },
                    train_cfg.best_ckpt_out,
                )

    Path(train_cfg.last_ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
            "step": train_cfg.train_steps - 1,
            "val_exact": best_val,
            "params": params,
        },
        train_cfg.last_ckpt_out,
    )

    summary = {
        "run_name": train_cfg.run_name,
        "params": params,
        "best_val_exact": best_val,
        "best_step": best_step,
        "train_steps": train_cfg.train_steps,
        "elapsed_sec": time.time() - t0,
        "metrics_csv": str(metrics_path),
        "best_ckpt": str(train_cfg.best_ckpt_out),
        "last_ckpt": str(train_cfg.last_ckpt_out),
    }
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "config.json", {"model": asdict(model_cfg), "train": asdict(train_cfg)})
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Train smallest addition transformer")

    # run/output
    p.add_argument("--run-name", type=str, default="repro_l1_d8_ff12")
    p.add_argument("--run-dir", type=Path, default=Path("results/runs/repro_l1_d8_ff12"))
    p.add_argument("--split-dir", type=Path, default=Path("results/data"))
    p.add_argument("--best-ckpt-out", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--last-ckpt-out", type=Path, default=Path("checkpoints/last.pt"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=123)

    # model
    p.add_argument("--n-layer", type=int, default=1)
    p.add_argument("--d-model", type=int, default=8)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=12)
    p.add_argument("--dropout", type=float, default=0.0)

    # optimization
    p.add_argument("--train-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=300)

    # eval split
    p.add_argument("--val-size", type=int, default=2000)
    p.add_argument("--test-size", type=int, default=10000)
    p.add_argument("--eval-batch-size", type=int, default=512)

    args = p.parse_args()

    model_cfg = ModelConfig(
        n_layer=args.n_layer,
        d_model=args.d_model,
        n_head=args.n_head,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=INPUT_LEN,
        vocab_size=VOCAB_SIZE,
    )

    train_cfg = TrainConfig(
        seed=args.seed,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        val_size=args.val_size,
        test_size=args.test_size,
        eval_batch_size=args.eval_batch_size,
        run_name=args.run_name,
        run_dir=str(args.run_dir),
        split_dir=str(args.split_dir),
        best_ckpt_out=str(args.best_ckpt_out),
        last_ckpt_out=str(args.last_ckpt_out),
        device=args.device,
    )

    summary = train(model_cfg, train_cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
