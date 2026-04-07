# Phase 2a — GRU End-to-End Neural Decoder (N1)

**Project:** EE597 Search-Designed Trellis Codes with Neural Decoding  
**Author:** Abhishek Manjunath  
**Phase status:** Implementation complete; training not yet run  
**Primary file:** `neural_decoder.py`

---

## Table of Contents

1. [Overview and Goal](#1-overview-and-goal)
2. [System Context (Inherited from Phase 1)](#2-system-context-inherited-from-phase-1)
3. [Architecture](#3-architecture)
4. [Configuration (TrainConfig)](#4-configuration-trainconfig)
5. [Training Method](#5-training-method)
6. [Inference / Evaluation API](#6-inference--evaluation-api)
7. [Compute Budget Analysis](#7-compute-budget-analysis)
8. [Assumptions](#8-assumptions)
9. [Tests](#9-tests)
10. [Output Artifacts](#10-output-artifacts)
11. [Success Criteria](#11-success-criteria)
12. [Generalization Tests (Post-Training)](#12-generalization-tests-post-training)
13. [Key References](#13-key-references)
14. [Implementation Checklist](#14-implementation-checklist)

---

## 1. Overview and Goal

Phase 2a implements **N1: the Bidirectional GRU End-to-End Decoder**, which replaces the entire Viterbi algorithm with a learned recurrent neural network. The decoder takes the raw received signal and outputs bit probability estimates directly, without any explicit trellis structure.

**What N1 must achieve:**
- Beat B1 (Mismatched Viterbi) by **≥ 0.5 dB at BLER = 10⁻³**
- Operate within the **2.1M operation budget**
- Show **stable training** (monotically decreasing validation BLER)
- Generalize to **channel parameters outside training distribution**

---

## 2. System Context (Inherited from Phase 1)

All of the following are fixed and must not be changed without explicit instruction.

| Parameter | Value |
|-----------|-------|
| Modulation | BPSK ∈ {−1, +1} |
| Info bits | K = 256 |
| Constraint length | K_c = 7 (memory m = 6, states S = 2^m = 64) |
| Tail bits | m = 6 (zero-forced termination) |
| Total input bits | K + m = 256 + 6 = 262 |
| Coded block length | N = 2 × 262 = **524** coded bits (including 12 tail-bit coded bits) |
| Code rate | R = 1/2 per trellis section; R_eff = 256/524 ≈ 0.4885 |
| Trellis | NASA K=7, generators 171/133 (octal), d_free = 10 |
| SNR range | 0–10 dB |
| INR range (training) | −5 to 10 dB |
| INR range (eval) | −5 to 15 dB |
| Interference model | i[t] = A·sin(2πt/P + φ), P ∈ [8,32], φ ∈ [0,2π] |
| Amplitude | A = √(2 · INR_lin · σ²) |
| BLER eval target | BLER = 10⁻³ |
| Op budget | 2.1M operations per block |
| MC trials (eval) | 100,000 |

### Channel model

The received signal at time step t:

```
r[t] = x[t] + n[t] + i[t]
```

where:
- `x[t] ∈ {−1, +1}` — BPSK modulated coded bit
- `n[t] ~ N(0, σ²)` — i.i.d. AWGN, σ² = 1 / SNR_lin
- `i[t] = A · sin(2πt/P + φ)` — sinusoidal interference

---

## 3. Architecture

### 3.1 BiRNNDecoder (class in `neural_decoder.py`)

The decoder is a **Bidirectional RNN** (default: GRU) that processes the full received sequence of length 524 and outputs 256 info bit logits.

```
Input:  (batch, 524, 1) — raw received signal, one scalar per time step
BiGRU:  h=32/dir → concat → (batch, 524, 64)
Output: Linear(64, 1) shared across all 524 positions → (batch, 524)
Select: even positions 0, 2, 4, ..., 510 → (batch, 256) info bit logits
```

**Info bit positions:** For a rate-1/2 code, info bits map to even-indexed coded positions `[0, 2, 4, ..., 510]` — the first 512 coded symbols. The last 12 positions (512–523) correspond to tail bit encoding and are **excluded** from loss and output.

### 3.2 GRU Cell Equations (per direction, per time step)

```
Reset gate:     r_t = sigmoid( W_r · [h_{t-1}, x_t] + b_r )
Update gate:    z_t = sigmoid( W_z · [h_{t-1}, x_t] + b_z )
Candidate:      h̃_t = tanh( W_h · [r_t ⊙ h_{t-1}, x_t] + b_h )
New state:      h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

Where `⊙` is element-wise multiplication and `[·, ·]` is concatenation.

### 3.3 Output Layer

At each position t, the concatenated hidden state `c[t]` (64 dims from both forward and backward directions) maps to one logit:

```
logit[t] = W_out · c[t] + b_out
```

`W_out` is shape `(1, 64)`, shared across all 524 positions (weight sharing). The final output selects the 256 info bit positions.

### 3.4 Parameter Count (h=32, input_dim=1, bidirectional)

```
Per GRU direction: 3 gates × (h × (h + 1) + h) = 3 × (32×33 + 32) = 3 × 1056 = 3,168
Wait — actual formula: 3 gates × (h × (h + d) + h) where d=1
  = 3 × (32 × 33 + 32) = 3 × 1,088 = 3,264 per direction
Two directions: 6,528
Output layer: 64 × 1 + 1 = 65
Total: ~6,593 trainable parameters
```

### 3.5 RNN Cell Variants

Three variants are compared at matched compute budgets:

| Variant | Hidden/dir | Approx ops | Purpose |
|---------|-----------|------------|---------|
| Bidirectional GRU | 32 | ~6.6M (default, exceeds budget) | Primary candidate |
| Bidirectional LSTM | 26 | comparable | Has cell state for tracking periodicity |
| Bidirectional vanilla RNN | 32 | ~2.2M | Sanity check; expected to fail on length-524 sequences |

> **Note:** The default h=32 BiGRU at N=524 gives ~6.6M ops, which exceeds the 2.1M budget. Architecture trimming (reduce hidden size) is required before final evaluation — see §7.

### 3.6 Optional BatchNorm

`BatchNorm1d` can be inserted between the GRU output and the linear layer via `use_batchnorm=True` in `TrainConfig`. Off by default. Applied to the feature dimension (transpose, normalize, transpose back).

---

## 4. Configuration (TrainConfig)

Defined as a `@dataclass` in `neural_decoder.py`. All values are set here — no global state.

```python
@dataclass
class TrainConfig:
    # ── Architecture ──────────────────────────
    hidden_size: int = 32          # hidden units per direction
    input_dim: int = 1             # input feature dim (raw scalar)
    cell_type: str = "GRU"         # "GRU" | "LSTM" | "RNN"
    bidirectional: bool = True
    use_batchnorm: bool = False

    # ── Optimizer ─────────────────────────────
    lr: float = 1e-3               # initial learning rate
    lr_step_epoch: int = 50        # epoch at which LR drops
    lr_step_factor: float = 0.1    # multiplier: 1e-3 → 1e-4
    grad_clip: float = 1.0         # gradient clipping norm
    batch_size: int = 128          # blocks per mini-batch

    # ── Training schedule ─────────────────────
    batches_per_epoch: int = 1000  # batches per epoch
    n_epochs: int = 100            # max epochs
    pool_size: int = 50000         # pre-encoded codeword pool size

    # ── Validation ────────────────────────────
    val_blocks: int = 10000        # blocks for validation BLER
    val_snr_db: float = 5.0        # fixed validation SNR
    val_inr_db: float = 5.0        # fixed validation INR
    patience: int = 20             # early stopping patience

    # ── Channel training distributions ────────
    snr_range: tuple = (0.0, 10.0)        # dB, uniform
    inr_range: tuple = (-5.0, 10.0)       # dB, uniform (note: 10 not 15 — generalization test)
    period_range: tuple = (8, 32)          # integer, uniform

    # ── Paths / device ─────────────────────────
    checkpoint_dir: str = ""     # default: results/phase2a/checkpoints/
    log_dir: str = ""            # default: results/phase2a/logs/
    device: str = ""             # default: "cuda" if available else "cpu"
```

**Key design decisions:**
- `inr_range` upper bound is **10 dB** during training (not 15 dB). This deliberately leaves out 10–15 dB to test **out-of-distribution generalization**.
- `pool_size = 50000` pre-encoded codewords for efficiency — avoids re-encoding on every batch.
- `batches_per_epoch = 1000` (reduced from the 10,000 in CLAUDE.md spec due to pool-based approach), giving 1000 × 128 = 128,000 training blocks per epoch.

---

## 5. Training Method

### 5.1 Data Generation Pipeline

Training uses **on-the-fly channel randomization** with **pre-encoded codeword pool**:

1. **Build pool** (`build_codeword_pool`): Pre-encode `pool_size` random info blocks through the NASA K=7 trellis and BPSK modulate. Store as `(pool_size, K_INFO)` info bits and `(pool_size, 524)` symbols.

2. **Per mini-batch** (`generate_training_batch`):
   - Sample `batch_size` random indices from pool
   - Per block, sample fresh channel parameters:
     - `SNR ~ Uniform(0, 10) dB`
     - `INR ~ Uniform(-5, 10) dB`
     - `P ~ Uniform_int(8, 32)` (integer periods)
     - `φ ~ Uniform(0, 2π)`
   - Generate AWGN noise and sinusoidal interference via `channel.py`
   - Add to symbol: `received[i] = symbol[i] + noise + interference`
   - Return `(batch, 524, 1)` received tensor and `(batch, 256)` info bit labels

3. **Loss:** Binary Cross-Entropy With Logits (BCEWithLogitsLoss):
   ```
   L = -(1/K) Σ_{k=0}^{K-1} [ b[k]·log σ(logit[k]) + (1−b[k])·log(1 − σ(logit[k])) ]
   ```
   where K = 256, b[k] ∈ {0,1} is the true bit, logit[k] is the model output.

   BCE is chosen because it is smooth and differentiable (required for gradient-based training). BLER is **not** used for training — it is non-differentiable and only computed during evaluation.

4. **Optimizer:** Adam
   - LR schedule: `StepLR(step_size=50, gamma=0.1)` → LR decays from 1e-3 to 1e-4 at epoch 50
   - Gradient clipping: `clip_grad_norm_(model.parameters(), 1.0)` — prevents exploding gradients in RNNs

### 5.2 Validation Protocol

Every epoch:
- Evaluate BLER on `val_blocks = 10,000` blocks
- Fixed point: `SNR = 5 dB`, `INR = 5 dB`
- Period P and phase φ are re-randomized (P ∈ [8,32])
- Metric: BLER = (blocks with ≥1 bit error) / total blocks
- Prediction: `bit = (sigmoid(logit) > 0.5)`

**Early stopping:** If validation BLER does not improve for `patience=20` consecutive epochs, training halts.

### 5.3 Checkpointing

Checkpoint saved whenever validation BLER improves. Saved to `results/phase2a/checkpoints/best_{cell_type}.pt`.

Checkpoint contains:
```python
{
    'model_state_dict': model.state_dict(),
    'config': asdict(config),       # full TrainConfig
    'epoch': epoch,
    'val_bler': val_bler,
    'loss': avg_loss,
}
```

### 5.4 Training Log

Per-epoch JSON log saved to `results/phase2a/logs/training_{cell_type}_h{hidden_size}.json`:
```json
[
  {"epoch": 0, "loss": 0.693, "val_bler": 0.72, "lr": 0.001, "time_s": 12.4},
  ...
]
```

### 5.5 Training Scale

| Quantity | Value |
|---------|-------|
| Batches per epoch | 1,000 |
| Batch size | 128 |
| Training blocks per epoch | 128,000 |
| Max epochs | 100 |
| Total max training blocks | 12.8M |
| Validation blocks per epoch | 10,000 |

### 5.6 Entry Point

```bash
# Train default GRU
python neural_decoder.py

# Or in code:
from neural_decoder import TrainConfig, train_model
config = TrainConfig(cell_type="GRU", hidden_size=32)
model = train_model(config, seed=42)
```

---

## 6. Inference / Evaluation API

### 6.1 make_decoder_n1

Returns a `decode_fn` compatible with `eval.py`'s `estimate_bler`:

```python
decode_fn = make_decoder_n1(model, device="cpu")

# Signature:
def decode_fn(received: np.ndarray,    # shape (524,), raw received signal
              period: float,
              phase: float,
              snr_db: float,
              inr_db: float) -> np.ndarray:  # shape (256,), dtype int8
```

The decode function:
1. Pads received signal to length 524 if shorter
2. Reshapes to `(1, 524, 1)` tensor
3. Runs forward pass through model
4. Applies sigmoid and thresholds at 0.5

**Note:** The N1 decoder does **not** use `period`, `phase`, `snr_db`, or `inr_db` — these arguments are present only for API compatibility with `eval.py`. The model implicitly infers channel conditions from the raw received signal.

### 6.2 load_model

```python
from neural_decoder import load_model
model = load_model("results/phase2a/checkpoints/best_gru.pt", device="cpu")
```

Reconstructs `BiRNNDecoder` from checkpoint config dict — no need to specify architecture separately.

---

## 7. Compute Budget Analysis

Budget: **2.1M operations per block**.

### 7.1 Analytical formula (in `compute_cost.py`)

```python
def count_flops_birnn_analytical(hidden_size, input_dim, seq_len, cell_type, bidirectional):
    ops_per_unit = {"GRU": 6, "LSTM": 8, "RNN": 2}[cell_type]
    n_dir = 2 if bidirectional else 1
    rnn_flops = n_dir * seq_len * ops_per_unit * hidden_size * (hidden_size + input_dim)
    linear_flops = n_dir * hidden_size * K_INFO  # output layer
    return rnn_flops + linear_flops
```

### 7.2 Op counts at various hidden sizes (BiGRU, d=1, N=524)

| Hidden size h | Approx ops | % of 2.1M budget | Notes |
|--------------|-----------|-----------------|-------|
| 32 | ~6.63M | 316% | **Default; exceeds budget 3×** |
| 20 | ~2.52M | 120% | Still over budget |
| 16 | ~1.62M | 77% | Under budget |
| 18 | ~2.04M | 97% | Under budget |
| 17 | ~1.83M | 87% | Under budget |

> **Action required before final evaluation:** Use `count_flops_birnn_analytical` to sweep h and find the exact hidden size fitting within 2.1M ops. The test `test_find_budget_hidden_size` in `test_neural_ops.py` does this automatically.

### 7.3 Budget adjustment options (from CLAUDE.md)

- **Option (a):** Reduce h to ~16–18 per direction
- **Option (b):** Process in temporal chunks of length 128 with overlap
- **Option (c):** Use unidirectional GRU h=40 (~2.0M ops); loses future context

The test `test_neural_ops.py::TestAnalyticalOps::test_bigru_h32_ops` documents that h=32 gives ~6.6M ops and is marked as acceptable for initial training — budget trimming is a separate step.

---

## 8. Assumptions

### 8.1 Architectural assumptions

1. **Input is scalar:** Each time step receives a single float `r[t]` (the raw received BPSK+noise+interference sample). There is no explicit time index or channel state provided as input — the model must infer all information from the received sequence alone.

2. **Rate-1/2 encoding:** Info bits map to even-indexed coded positions `{0, 2, 4, ..., 510}`. This is hardcoded in `BiRNNDecoder.info_positions` as `torch.arange(0, 2*K_INFO, 2)`.

3. **Tail bit positions excluded from loss:** The 12 coded bits at positions 512–523 (from the 6 tail bits appended for trellis termination) are not included in the BCE loss computation or output selection.

4. **Shared output weights:** The same `Linear(64, 1)` layer processes each of the 524 time steps. This is a form of weight sharing that reduces parameters and enforces time-translation equivariance.

5. **Logits, not probabilities, returned by forward pass:** `model(x)` returns raw logits. `sigmoid` is applied externally (in `make_decoder_n1`) or implicitly within `BCEWithLogitsLoss`. This is numerically more stable than computing `BCE(sigmoid(logits), targets)`.

### 8.2 Training distribution assumptions

6. **Training INR upper bound is 10 dB, not 15 dB:** The model trains on `INR ∈ [−5, 10]` dB. Evaluation at INR = 12 dB and 15 dB tests **out-of-distribution generalization**. This is intentional: a good model should degrade gracefully, not catastrophically.

7. **Integer periods:** `P` is sampled as an integer from `[8, 32]`. The true interference uses exactly integer periods. Experiment note: CLAUDE.md suggests also trying continuous period sampling.

8. **Phase is fresh per block:** Phase `φ` is re-sampled uniformly for every training block and every evaluation block. The model sees no two blocks with the same phase.

9. **Pool-based encoding for efficiency:** Pre-encoding 50,000 codewords avoids running `trellis.encode()` inside the training hot loop. Channel randomization (noise + interference) is still done fresh per batch.

### 8.3 Evaluation assumptions

10. **BLER primary metric:** A block is in error if **any** of the 256 decoded bits differs from the true info bits. Bit error rate (BER) is only for debugging.

11. **100,000 Monte Carlo blocks per evaluation point:** Required for reliable BLER estimates at 10⁻³. The validation set during training uses only 10,000 blocks for speed.

12. **Symmetric training and evaluation:** Both use `channel.py`'s `generate_interference`, `noise_var_from_snr`, and `amplitude_from_inr`. No mismatched channel assumptions in N1.

13. **No test-time oracle:** N1 does not know the true `P`, `φ`, or `A` during decoding. These are only implicitly estimated from the received signal via the GRU's learned hidden representations.

### 8.4 Compute measurement assumptions

14. **Analytical FLOPs (multiply-adds):** Op count uses the formula `n_dir × N × 6×h×(h+d)` for GRU. This counts multiply-accumulates (MACs), not FLOPs (each MAC = 2 FLOPs). The budget convention (MACs vs FLOPs) must be consistent across all methods — `compute_cost.py` enforces this.

15. **Sequence length is 524 (not 512):** Due to 12 termination coded bits. Ops scale linearly with N, so this matters for budget calculations.

---

## 9. Tests

Three test files in `tests/` cover Phase 2a:

### 9.1 `test_neural_ops.py` — Architecture verification

| Test | What it checks | Threshold |
|------|---------------|-----------|
| `TestOutputShape::test_single_sample` | Forward pass output shape (1, 256) | Exact |
| `TestOutputShape::test_batch` | Batch forward pass (16, 256) | Exact |
| `TestOutputShape::test_output_is_logits` | Output contains values < 0 or > 1 (not probabilities) | Structural |
| `TestParameterCount::test_bigru_h32_param_count` | ~6,593 parameters | [6000, 7500] |
| `TestAnalyticalOps::test_bigru_h32_ops` | ~6.6M ops for h=32 | [6M, 7.5M] |
| `TestAnalyticalOps::test_find_budget_hidden_size` | Print max h fitting 2.1M budget | Informational |

### 9.2 `test_neural_overfit.py` — Training sanity check

| Test | What it checks | Threshold |
|------|---------------|-----------|
| `test_overfit_single_batch` | Model memorizes 4 blocks at SNR=10dB, INR=0dB after 500 steps | `loss < 0.4`, `accuracy > 0.85` |
| `test_loss_decreases` | Loss decreases over 200 steps | `last_20 < first_20 * 0.8` |

**Setup:** Batch of 4 blocks, SNR=10 dB, INR=0 dB (nearly interference-free), period=16, LR=5e-3.

### 9.3 `test_neural_vs_b1.py` — Performance gating (requires trained checkpoint)

| Test | What it checks | Threshold |
|------|---------------|-----------|
| `test_n1_beats_b1_at_5db` | N1 BLER < B1 BLER at SNR=5dB, INR=5dB | N1 < B1 (any improvement) |
| `test_n1_bler_decreases_with_snr` | BLER at 3, 5, 7 dB is monotonically decreasing | allows ±0.02 tolerance |

**Note:** These tests are marked `@pytest.mark.slow` and `@requires_checkpoint`. They are skipped if `results/phase2a/checkpoints/best_gru.pt` does not exist.

### 9.4 Running tests

```bash
# Smoke tests (fast, no checkpoint needed)
pytest tests/test_neural_ops.py -v
pytest tests/test_neural_overfit.py -v

# Performance gate (requires trained checkpoint)
pytest tests/test_neural_vs_b1.py -v -m slow
```

---

## 10. Output Artifacts

All saved to `results/phase2a/`:

```
results/phase2a/
├── checkpoints/
│   ├── best_gru.pt      # Best GRU model (lowest val BLER)
│   ├── best_lstm.pt     # Best LSTM model
│   └── best_rnn.pt      # Best vanilla RNN model
├── figures/
│   ├── phase2a_bler_vs_snr.pdf   # BLER vs SNR at INR=5dB: N1, B1, B2, B5
│   ├── phase2a_bler_vs_inr.pdf   # BLER vs INR at SNR=5dB: N1, B1, B2, B5
│   └── training_loss.pdf         # BCE loss + val BLER vs epoch
└── logs/
    └── training_gru_h32.json     # Per-epoch: loss, val_bler, lr, time_s
```

---

## 11. Success Criteria

From CLAUDE.md:

| Criterion | Threshold |
|-----------|-----------|
| N1 vs B1 at BLER = 10⁻³ | **N1 must beat B1 by ≥ 0.5 dB** |
| Training stability | Val BLER decreases monotonically (or is not worse) across epochs |
| Compute budget | Final architecture ops ≤ 2.1M per block |
| Overfit test | Model can memorize 4 training blocks (loss < 0.4, accuracy > 0.85) |

---

## 12. Generalization Tests (Post-Training)

After training, evaluate at out-of-distribution channel conditions:

| Test | In-distribution | OOD condition | What we want |
|------|----------------|--------------|--------------|
| High INR | INR ≤ 10 dB | INR = 12, 15 dB | Graceful degradation, not cliff |
| Short period | P ≥ 8 | P = 4 | Partial degradation acceptable |
| Long period | P ≤ 32 | P = 48 | Likely fails; expected |
| SNR shift | SNR ∈ [0,10] | SNR ∈ [5,15] | Should still decode at high SNR |

These generalization tests distinguish a model that has genuinely learned channel structure from one that has merely memorized training statistics.

---

## 13. Key References

| Paper | Relevance |
|-------|-----------|
| Gruber et al., "On Deep Learning-Based Channel Decoding" (2017) | Foundational RNN decoder for convolutional codes |
| Jiang et al., "LEARN Codes: Inventing Low-Latency Codes via RNNs" (2019, IEEE TCOM) | GRU encoder/decoder co-design, robustness under channel mismatch |
| Kim et al., "Communication Algorithms via Deep Learning" (2018) | Bi-GRU decoder with BatchNorm, reference implementation |
| IEEE ICC 2024, "Deep Learning Based Decoder for Concatenated Coding over Deletion Channels" | Bi-GRU as LLR estimator, single network across channel params |

---

## 14. Implementation Checklist

### Completed ✅

- [x] `BiRNNDecoder(nn.Module)` with configurable `hidden_size`, `cell_type`, `bidirectional`, `use_batchnorm`, `input_dim`
- [x] `TrainConfig` dataclass with all hyperparameters and automatic path resolution
- [x] `build_codeword_pool(trellis, pool_size, seed)` — pre-encodes codewords, handles variable-length symbols
- [x] `generate_training_batch(pool, batch_size, snr_range, inr_range, period_range, rng, device)` — fresh channel per block
- [x] `validate_bler(model, trellis, pool, snr_db, inr_db, n_blocks, device, seed)` — BLER evaluation without gradient
- [x] `train_model(config, seed)` — full training loop with Adam + StepLR + gradient clipping + early stopping + checkpointing + JSON logging
- [x] `make_decoder_n1(model, device)` — `eval.py`-compatible decode wrapper
- [x] `load_model(checkpoint_path, device)` — reconstruct from checkpoint
- [x] Smoke test in `__main__` block
- [x] `test_neural_ops.py` — architecture and op count tests
- [x] `test_neural_overfit.py` — training sanity check
- [x] `test_neural_vs_b1.py` — performance gating (skips without checkpoint)

### Pending ⬜

- [ ] Find budget-compliant hidden size via `test_find_budget_hidden_size` and retrain with it
- [ ] Train GRU variant: `train_model(TrainConfig(cell_type="GRU"), seed=42)`
- [ ] Train LSTM variant: `train_model(TrainConfig(cell_type="LSTM", hidden_size=26), seed=42)`
- [ ] Train vanilla RNN variant: `train_model(TrainConfig(cell_type="RNN"), seed=42)`
- [ ] Evaluate: BLER vs SNR sweep at INR=5 dB (100,000 MC blocks per point)
- [ ] Evaluate: BLER vs INR sweep at SNR=5 dB (100,000 MC blocks per point)
- [ ] Generalization tests: INR=12, 15 dB; P=4, 48
- [ ] Compare N1 vs B1, B2, B5 — verify ≥0.5 dB gain at BLER=10⁻³
- [ ] Generate `phase2a_bler_vs_snr.pdf`, `phase2a_bler_vs_inr.pdf`, `training_loss.pdf`
- [ ] Report measured op count from `count_flops_birnn_analytical` (or `thop`) in `compute_cost.py`
- [ ] Populate Phase 2a row in the compute table in the progress report

---

*Last updated: April 6, 2026*
