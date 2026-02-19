# Smallest Transformer for 10-Digit Addition (>=99% EM)

## 1) Final Best Model (smallest found that passed)

- **Run name:** `search_l1_d8_ff12`
- **Layers:** 1 decoder block
- **Hidden dim (`d_model`):** 8
- **Heads:** 2
- **FFN dim (`d_ff`):** 12
- **Dropout:** 0.0
- **Context length (`max_seq_len`):** 23 tokens
- **Vocabulary size:** 114 tokens
- **Total parameters:** **1,644**

### Token inventory
- Output digit tokens: `0..9` (10)
- Input pair tokens: `P00..P99` (100)
- Specials: `=`, `<bos>`, `<eos>`, `<pad>` (4)

Total vocab = 10 + 100 + 4 = 114.

## 2) Data Pipeline

### Preprocess
Deterministic `preprocess(A,B) -> model_input`:

1. Zero-pad both to 10 digits.
2. Reverse each to LSD->MSD digit order: `a0..a9`, `b0..b9`.
3. Map each column `(a_i, b_i)` to one token `P{a_i}{b_i}`.
4. Prompt format:
   - `<bos> P(a0,b0) P(a1,b1) ... P(a9,b9) =`

This keeps all column information explicit while leaving carry propagation to the transformer.

### Target format
- Output sequence (teacher-forced in training, autoregressive at inference):
  - `c0 c1 ... c10 <eos>`
  - where `c0..c10` are digits of `A+B` in **reversed** order (LSD->MSD), fixed width 11.

### Postprocess
Deterministic `postprocess(model_output) -> C`:

1. Read generated tokens until `<eos>` (or first non-digit).
2. Keep first 11 digit tokens (pad with zeros if too short).
3. Reverse back to normal order and parse integer.

## 3) Training Details

- **Framework:** PyTorch 2.5.1 (CPU)
- **Objective:** Autoregressive cross-entropy
- **Loss mask:** only answer tokens (`11 digits + <eos>`) contribute to loss
- **Optimizer:** AdamW
- **LR schedule:** linear warmup + cosine decay
- **Hyperparameters (final run):**
  - `lr=3e-3`
  - `warmup_steps=100`
  - `min_lr_ratio=0.1`
  - `weight_decay=0.01`
  - `grad_clip=1.0`
- **Batch size:** 512
- **Train steps:** 5,000
- **Training examples seen:** 2,560,000
- **Validation:** every 300 steps on 2,000 held-out examples
- **Training time (final run):** ~95.4s

### Holdout/generalization safeguards
- Validation/test pairs are generated once from a fixed seed and saved.
- Training sampler checks sampled `(A,B)` against all holdout pairs and resamples on collision.
- Test set size is **10,000** as required.

## 4) Curves / Artifacts

- Training script: `experiment.py`
- Final run config: `artifacts/search_l1_d8_ff12/config.json`
- Final run metrics: `artifacts/search_l1_d8_ff12/metrics.csv`
- Curves plot (loss + val EM): `artifacts/search_l1_d8_ff12/curves.png`
- Final run summary: `artifacts/search_l1_d8_ff12/summary.json`
- Test evaluation: `artifacts/search_l1_d8_ff12/test_results.json`

## 5) Final Evaluation on 10,000 Held-Out Test Examples

Checkpoint: `artifacts/search_l1_d8_ff12/checkpoints/best.pt`

- **Exact-match accuracy:** **0.9904** (99.04%)
- **Per-token digit accuracy:** 0.9990727
- **Meets target >=99% exact-match:** **Yes**

### Failure samples (10)

1. `A=5494929602, B=5097590747, pred=10592510349, true=10592520349`
2. `A=2387190696, B=8725060082, pred=11112260778, true=11112250778`
3. `A=9930973050, B=9755128054, pred=9686101104, true=19686101104`
4. `A=8260033151, B=2937091401, pred=11197024552, true=11197124552`
5. `A=9310394203, B=9711475589, pred=9021869792, true=19021869792`
6. `A=8493050971, B=2211750872, pred=10704802843, true=10704801843`
7. `A=6999720508, B=4969121632, pred=11969842140, true=11968842140`
8. `A=6809322292, B=1649563090, pred=8459885382, true=8458885382`
9. `A=9466379984, B=894469793, pred=10360840777, true=10360849777`
10. `A=822370053, B=9449501863, pred=10271881916, true=10271871916`

## 6) Why this format worked

- Reversed output aligns generation direction with carry flow.
- Pair tokens `Pab` compactly expose each digit-column in one token, reducing required context integration.
- This enabled a **1-layer, 1.6k-parameter** model to clear the 99% bar.


---

## Full Experiment Log

# Research Log (Transparent Iteration History)

## Goal
Train the smallest autoregressive transformer that reaches >=99% exact-match on held-out 10-digit addition.

## Environment
- CPU-only PyTorch (no CUDA).
- All work and artifacts kept under this folder.

## Run-by-run history

### 1) Initial baseline format (raw reversed A and B digits)
- **Run:** `pilot_l2_d32`
- **Config:** 2 layers, d=32, h=4, ff=64
- **Params:** 18,720
- **Result:** `best val exact = 0.0000` (step 0), token acc stayed low.
- **Decision:** Format appeared hard for very small models under tight compute. Pivoted to a better tokenization.

### 2) Tokenization redesign: per-column pair token `Pab`
- **Reasoning:** Keep deterministic preprocess while making each addition column explicit and local.

#### 2a) Viability check
- **Run:** `pilot_pair_l2_d32`
- **Config:** 2 layers, d=32, h=4, ff=64
- **Params:** 21,536
- **Result:** crossed 99% quickly; reached `val exact = 1.0000` by step 800.
- **Decision:** Representation works; proceed to parameter minimization.

### 3) Aggressive minimization sweep

#### 3a) Very small 1-layer attempt
- **Run:** `search_l1_d8`
- **Config:** 1 layer, d=8, h=2, ff=16
- **Params:** 1,712
- **Result:** `best val exact = 0.9845` at step 2999 (close but below target).
- **Decision:** promising; test nearby sizes and longer schedules.

#### 3b) Too small hidden size
- **Run:** `search_l1_d6`
- **Config:** 1 layer, d=6, h=2, ff=12
- **Params:** 1,188
- **Result:** plateaued at `val exact = 0.0000` through observed checkpoints.
- **Decision:** insufficient capacity, aborted early.

#### 3c) Single-head d=7
- **Run:** `search_l1_d7_h1`
- **Config:** 1 layer, d=7, h=1, ff=14
- **Params:** 1,442
- **Result:** improved slowly, but only `val exact = 0.1910` best (step 3000).
- **Decision:** insufficient in this budget.

#### 3d) Longer run for d=8, ff=16
- **Run:** `search_l1_d8_long`
- **Config:** 1 layer, d=8, h=2, ff=16
- **Params:** 1,712
- **Result:** reached `val exact = 1.0000` by step 1800.
- **Decision:** confirms d=8/ff16 is strong; try smaller ff for fewer params.

#### 3e) Reduce FF width to 8
- **Run:** `search_l1_d8_ff8`
- **Config:** 1 layer, d=8, h=2, ff=8
- **Params:** 1,576
- **Result:** only `val exact = 0.6875` by step 3900.
- **Decision:** likely under-capacity.

#### 3f) Reduce FF width to 12 (candidate)
- **Run:** `search_l1_d8_ff12`
- **Config:** 1 layer, d=8, h=2, ff=12
- **Params:** **1,644**
- **Result:** `val exact = 0.9905` at step 4500, `0.9935` at step 4999.
- **Decision:** passes target and smaller than ff16.

#### 3g) Boundary checks below ff12
- **Run:** `search_l1_d8_ff10` (1,610 params) -> best `val exact = 0.1705`.
- **Run:** `search_l1_d8_ff11` (1,627 params) -> best `val exact = 0.7335`.
- **Run:** `search_l1_d8_ff8_tuned` (1,576 params, longer+tuned LR) -> still near 0% exact over observed checkpoints.
- **Decision:** boundary appears between ff11 and ff12 for this setup.

## Final selection rationale
Selected `search_l1_d8_ff12` because:
- It is the smallest architecture observed to exceed 99% validation EM.
- Smaller neighboring variants (ff11, ff10, ff8) consistently failed under extended training attempts.

## Final held-out test
- **Checkpoint:** `artifacts/search_l1_d8_ff12/checkpoints/best.pt`
- **Test size:** 10,000
- **Exact-match:** 99.04% (`0.9904`)
- **Outcome:** target achieved.
