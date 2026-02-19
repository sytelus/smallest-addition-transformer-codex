# smallest-addition-transformer-codex

**Result:** 1,644-parameter decoder-only transformer, **99.04% exact-match** on a 10,000-example held-out 10-digit addition test set.

## Quick Repro

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train from scratch (same final config used for best small model):

```bash
python -m src.train \
  --run-name repro_l1_d8_ff12 \
  --run-dir results/runs/repro_l1_d8_ff12 \
  --best-ckpt-out checkpoints/best.pt \
  --last-ckpt-out checkpoints/last.pt \
  --seed 123 \
  --n-layer 1 --d-model 8 --n-head 2 --d-ff 12 \
  --train-steps 5000 --batch-size 512 \
  --lr 3e-3 --warmup-steps 100 --min-lr-ratio 0.1 \
  --eval-interval 300 --val-size 2000 --test-size 10000 \
  --device cpu
```

Evaluate on held-out test set:

```bash
python -m src.eval test \
  --ckpt checkpoints/best.pt \
  --seed 123 --val-size 2000 --test-size 10000 \
  --eval-batch-size 512 --device cpu \
  --out-json results/final_results.json
```

Single-example inference:

```bash
python -m src.eval predict --ckpt checkpoints/best.pt --a 1234567890 --b 9876543210 --device cpu
```

## Approach (brief)

- **Tokenization:** pair-column tokens `Pab` for each digit column `(a_i, b_i)` in LSD->MSD order.
- **Prompt format:** `<bos> P(a0,b0) ... P(a9,b9) =`
- **Target format:** reversed 11-digit sum `c0..c10` + `<eos>`.
- **Model:** tiny GPT-style decoder with weight tying.
- **Key idea:** pair tokens expose per-column addition structure directly, reducing alignment burden and enabling very small models.

## Reports

- Full markdown report: [REPORT.md](REPORT.md)
- PDF report: [report.pdf](report.pdf)
- AI-agent handoff notes: [HANDOFF.md](HANDOFF.md)
