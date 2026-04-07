# Phase 2a — GRU End-to-End Neural Decoder (N1)

**Project:** EE597 Search-Designed Trellis Codes with Neural Decoding  
**Author:** Abhishek Manjunath  
**Phase status:** Training experiments complete — **negative result** (architecture cannot learn end-to-end decoding)  
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
10. [Experimental Results](#10-experimental-results)
11. [Output Artifacts](#11-output-artifacts)
12. [Success Criteria](#12-success-criteria)
13. [Phase 2a Conclusion and Motivation for Phase 2b](#13-phase-2a-conclusion-and-motivation-for-phase-2b)
14. [Generalization Tests (Post-Training)](#14-generalization-tests-post-training)
15. [Key References](#15-key-references)
16. [Implementation Checklist](#16-implementation-checklist)

---

## 1. Overview and Goal

Phase 2a implements **N1: the Bidirectional GRU End-to-End Decoder**, which replaces the entire Viterbi algorithm with a learned recurrent neural network. The decoder takes the raw received signal and outputs bit probability estimates directly, without any explicit trellis structure.

**What N1 must achieve:**
- Beat B1 (Mismatched Viterbi) by **≥ 0.5 dB at BLER = 10⁻³**
- Operate within the **2.1M operation budget**
- Show **stable training** (monotonically decreasing validation BLER)
- Generalize to **channel parameters outside training distribution**

### 1.1 Phase 2a Outcome Summary

**Result: Negative.** The BiGRU end-to-end decoder cannot learn to decode the rate 1/2 K=7 convolutional code. After extensive experimentation — including curriculum training, multiple pooling strategies (mean pooling, attention pooling), and training at very high SNR (15–20 dB) with near-zero interference — the model plateaued at ~53% bit accuracy. Random guessing is 50%. BLER remained at 1.0 throughout all experiments. None of the success criteria were met.

This negative result provides strong motivation for Phase 2b, where the same BiGRU architecture is repurposed for branch metric estimation rather than end-to-end decoding. The Viterbi algorithm handles the global trellis search; the GRU handles local channel estimation.

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
It is Bidirectional because the received sequence is processed in forward direction (t=0 to t=523) and backward direction(t=523 to t=0)
```
Input:   (batch, 524, 1) — raw received signal, one scalar per time step
BiGRU:   h=32/dir → concat → (batch, 524, 64)
Pool:    Mean over time dimension → (batch, 64)
Output:  Linear(64, 256) → (batch, 256) info bit logits
```

**Mean-pooling design (baseline):** Rather than selecting specific coded positions, the GRU output is averaged over all 524 time steps to produce a single 64-dim context vector. A `Linear(64, 256)` layer then maps this directly to 256 info bit logits. An **attention pooling** variant was also tested, which learns a per-timestep importance weight instead of uniform averaging. See §3.3 for details on all pooling strategies and their results.

### 3.2 GRU Cell Equations (per direction, per time step)

```
Reset gate:     r_t = sigmoid( W_r · [h_{t-1}, x_t] + b_r )
Update gate:    z_t = sigmoid( W_z · [h_{t-1}, x_t] + b_z )
Candidate:      h̃_t = tanh( W_h · [r_t ⊙ h_{t-1}, x_t] + b_h )
New state:      h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

Where `⊙` is element-wise multiplication and `[·, ·]` is concatenation.

### 3.3 Output Layer and Pooling Strategies

The GRU produces a per-timestep output of shape `(batch, 524, 2H)`. This must be reduced to `(batch, 256)` info bit logits. Three pooling strategies were implemented and tested.

#### 3.3.1 Mean Pooling (baseline)

The GRU output is averaged over all 524 timesteps to produce a single context vector `c` (64 dims), which is then mapped to all 256 info bit logits:

```
c = (1/N) Σ_{t=0}^{N-1} rnn_out[t]     # mean-pool: (batch, 64)
logits = W_out · c + b_out              # Linear(64, 256): (batch, 256)
```

`W_out` is shape `(256, 64)` — one row per info bit. Every timestep contributes equally. This is the weakest strategy because some timesteps carry more decoding information than others, and the uniform weighting dilutes the signal.

**Result:** Plateaued at 52.1% bit accuracy after 200 epochs at SNR 15–20 dB. See §10.


#### 3.3.2 Final Hidden States

A BiGRU inherently produces a sequence summary in its final hidden states. The forward direction's final state summarizes left-to-right context; the backward direction's final state summarizes right-to-left context. Concatenating both gives a (2H)-dim vector without any pooling.

```
rnn_out, h_n = self.rnn(x)              # h_n: (2, batch, H) for bidirectional
c = h_n.transpose(0,1).reshape(B, 2H)   # (batch, 2H)
logits = W_out · c + b_out              # Linear(2H, 256): (batch, 256)
```
**Result:** Plateaued at 53.5% bit accuracy after 200 epochs at SNR 15–20 dB. Slightly better performance that **mean pooling**

#### 3.3.3 Attention Pooling

Instead of uniform averaging, attention pooling learns a weight per timestep. The model decides which timesteps are most important for decoding.
A learned parameter vector `v` of size 2H acts as a query. For each timestep `t`, a relevance score is computed:

```
e_t = v · h_t                           # dot product: scalar per timestep
a_t = exp(e_t) / Σ_j exp(e_j)          # softmax: normalized weights summing to 1
c = Σ_t a_t · h_t                       # weighted sum: (batch, 2H)
logits = W_out · c + b_out              # Linear(2H, 256): (batch, 256)
```

The vector `v` is initialized randomly and updated by gradient descent during training, just like all other network weights. After training, `v` points in the direction of the 2H-dimensional GRU output space that best separates important timesteps from unimportant ones. Timesteps whose GRU outputs align with `v` receive high attention weights; those pointing away receive low weights. This adds only 64 new parameters (the vector `v`), bringing the total to 23,424.
**Result:** Plateaued at 53.8% bit accuracy after 200 epochs at SNR 15–20 dB. Marginal improvement over mean pooling. See §10.

### 3.4 Parameter Count (h=32, input_dim=1, bidirectional)

```
Per GRU direction: 3 gates × (h × (h + d) + h) where d=1
  = 3 × (32 × 33 + 32) = 3 × 1,088 = 3,264 per direction
Two directions: 6,528
Output layer: 64 × 256 + 256 = 16,640
BatchNorm (if enabled): 2 × 64 = 128 (off by default)
Total: ~23,360 trainable parameters (with BatchNorm disabled)
```

> **Note:** The output layer accounts for most parameters (16,640 / 23,360 ≈ 71%) due to the `Linear(64, 256)` projection. The GRU itself is lightweight at 6,528 parameters.

### 3.5 RNN Cell Variants

Three variants are compared at matched compute budgets:

| Variant | Hidden/dir | Parameters | Approx ops | Purpose |
|---------|-----------|-----------|------------|---------|
| Bidirectional GRU | 32 | 23,360 | ~6.6M (default, exceeds budget) | Primary candidate |
| Bidirectional LSTM | 26 | comparable | comparable | Has cell state for tracking periodicity |
| Bidirectional vanilla RNN | 32 | comparable | ~2.2M | Sanity check; expected to fail on length-524 sequences |

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
    val_blocks: int = 10000        # blocks for validation
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

### 4.1 Validation Configuration — Lessons Learned

**Critical lesson: validation SNR/INR must match training SNR/INR.**

The original default configuration validated at SNR=5 dB and INR=5 dB regardless of training conditions. When training at SNR=10 dB and INR=0 dB, this meant the model was being evaluated on a harder channel than it had ever seen during training. BLER stayed at 1.0 and gave no useful signal about whether the model was learning.

The fix: set `val_snr_db` and `val_inr_db` to the midpoint of the training range. When training over a range (e.g., SNR 0–10 dB), validate at the midpoint (SNR=5 dB). When training at a fixed point (e.g., SNR 15–20 dB), validate at the midpoint of that range (SNR=17.5 dB).

**Checkpointing metric: bit accuracy, not BLER.** BLER is binary — a block is either perfect or has errors. When the model is early in training at ~52% bit accuracy, every block has errors and BLER is stuck at 1.0. Bit accuracy is continuous and captures incremental progress. Checkpointing was changed to save on best validation bit accuracy.

### 4.2 Curriculum Training Configurations

Curriculum training starts with easy channel conditions and gradually increases difficulty. Three stages were defined:

```python
# Stage 1: Easy. High SNR, minimal interference.
config_stage1 = TrainConfig(
    snr_range=(15.0, 20.0),
    inr_range=(-5.0, -2.5),
    val_snr_db=17.5,
    val_inr_db=-3.75,
    n_epochs=200,
    patience=200,
)

# Stage 2: Medium. Widen both ranges.
config_stage2 = TrainConfig(
    snr_range=(4.0, 10.0),
    inr_range=(-5.0, 7.5),
    val_snr_db=7.0,
    val_inr_db=1.25,
    n_epochs=50,
    patience=50,
)

# Stage 3: Full range.
config_stage3 = TrainConfig(
    snr_range=(0.0, 10.0),
    inr_range=(-5.0, 15.0),
    val_snr_db=5.0,
    val_inr_db=5.0,
    n_epochs=100,
    patience=100,
)
```

**Note:** `train_model` was modified to accept an optional pretrained model parameter so that each stage resumes from the previous stage's checkpoint rather than initializing a fresh model. Stage 2 was attempted but the model had not learned enough in Stage 1 to benefit from curriculum progression. See §10 for details.

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
   - LR schedule: `StepLR(step_size=50, gamma=0.1)` → LR stays constant and optionally can decays by lr_step_factor at epoch % lr_step_epoch == 0
   - Gradient clipping: `clip_grad_norm_(model.parameters(), 1.0)` — disabled but can prevents exploding gradients in RNNs

### 5.2 Validation Protocol

Every epoch:
- Evaluate bit accuracy and BLER on `val_blocks = 10,000` blocks
- Fixed point: validation SNR/INR set to **midpoint of training range** (see §4.1)
- Period P and phase φ are re-randomized (P ∈ [8,32])
- Metrics:
  - Bit accuracy = (correctly decoded bits) / (total bits across all blocks)
  - BLER = (blocks with ≥1 bit error) / total blocks
- Prediction: `bit = (sigmoid(logit) > 0.5)`

**Early stopping:** If validation bit accuracy does not improve for `patience` consecutive epochs, training halts. (disabled)
### 5.3 Checkpointing

Checkpoint saved whenever **validation bit accuracy** improves. Saved to `results/phase2a/checkpoints/best_{cell_type}.pt`.

> **Lesson learned:** The original implementation checkpointed on best validation BLER. When the model is early in training at ~52% bit accuracy, every block has errors and BLER is stuck at 1.0. The first epoch always becomes the "best" checkpoint, and no subsequent improvements are captured. Bit accuracy is a continuous metric that captures incremental progress and is the correct metric for checkpointing during training.

Checkpoint contains:
```python
{
    'model_state_dict': model.state_dict(),
    'config': asdict(config),       # full TrainConfig
    'epoch': epoch,
    'val_bit_acc': val_bit_acc,
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

**Console output:** Epoch summaries are printed **every 50 epochs** (not every epoch) to reduce noise. Checkpoint save messages (`-> New best!`) and early stopping messages always print regardless of this interval.

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
    linear_flops = n_dir * hidden_size * K_INFO  # output layer: Linear(2H, K_INFO)
    return rnn_flops + linear_flops
```

### 7.2 Op counts at various hidden sizes (BiGRU, d=1, N=524)

| Hidden size h | Parameters | Approx ops | % of 2.1M budget | Notes |
|--------------|-----------|-----------|-----------------|-------|
| 32 | 23,360 | ~6.63M | 316% | **Default; exceeds budget 3×** |
| 20 | ~TBD | ~2.52M | 120% | Still over budget |
| 16 | ~TBD | ~1.62M | 77% | Under budget |
| 18 | ~TBD | ~2.04M | 97% | Under budget |
| 17 | ~TBD | ~1.83M | 87% | Under budget |

> **Action required before final evaluation:** Use `count_flops_birnn_analytical` to sweep h and find the exact hidden size fitting within 2.1M ops. The test `test_find_budget_hidden_size` in `test_neural_ops.py` does this automatically.

---

## 8. Assumptions

### 8.1 Architectural assumptions

1. **Input is scalar:** Each time step receives a single float `r[t]` (the raw received BPSK+noise+interference sample). There is no explicit time index or channel state provided as input — the model must infer all information from the received sequence alone.

2. **Pooling-based output:** The GRU output is reduced from 524 timesteps to a fixed-length context vector via pooling (mean or attention), then projected to 256 info bit logits via `Linear(2H, 256)`. There is no explicit mapping between coded positions and info bit positions — the model learns this mapping implicitly. Both mean pooling and attention pooling were tested; see §3.3.

3. **Tail bits handled implicitly:** The 12 coded bits at positions 512–523 (from tail bits) contribute to the mean pool and thus the global context, but there is no explicit exclusion — the model sees all 524 time steps equally.

4. **Global output projection:** A single `Linear(64, 256)` maps the pooled representation to all 256 info bit logits simultaneously. This replaces the earlier per-timestep `Linear(64, 1)` + position-selection design.

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
---

## 9. Tests

Four test files in `tests/` cover Phase 2a:

### 9.1 `test_neural_ops.py` — Architecture verification

| Test | What it checks | Threshold |
|------|---------------|-----------|
| `TestOutputShape::test_single_sample` | Forward pass output shape (1, 256) | Exact |
| `TestOutputShape::test_batch` | Batch forward pass (16, 256) | Exact |
| `TestOutputShape::test_output_is_logits` | Output contains values < 0 or > 1 (not probabilities) | Structural |
| `TestParameterCount::test_bigru_h32_param_count` | ~23,360 parameters | Should be verified |
| `TestAnalyticalOps::test_bigru_h32_ops` | ~6.6M ops for h=32 | [6M, 7.5M] |
| `TestAnalyticalOps::test_find_budget_hidden_size` | Print max h fitting 2.1M budget | Informational |

### 9.2 `test_neural_overfit.py` — Training sanity check (pytest)

| Test | What it checks | Threshold |
|------|---------------|-----------|
| `test_overfit_single_batch` | Model memorizes 4 blocks at SNR=10dB, INR=0dB after 500 steps | `loss < 0.4`, `accuracy > 0.85` |
| `test_loss_decreases` | Loss decreases over 200 steps | `last_20 < first_20 * 0.8` |

**Setup:** Batch of 4 blocks, SNR=10 dB, INR=0 dB (nearly interference-free), period=16, LR=5e-3.

### 9.3 `test_overfit_gru_batch.py` — Standalone overfit script

A standalone Python script (no pytest required) that overfits the BiGRU decoder on a single small batch and prints epoch-by-epoch loss. Designed for quick interactive sanity checking.

```bash
python tests/test_overfit_gru_batch.py
```

| Parameter | Value |
|-----------|-------|
| Batch size | 4 |
| Epochs | 500 |
| LR | 5e-3 (Adam) |
| SNR | 10.0 dB (fixed) |
| INR | 0.0 dB (fixed) |
| Period | 16 (fixed) |
| Print every | 50 epochs |

**Pass criteria:** Final loss < 0.05, loss dropped ≥ 50%, bit accuracy ≥ 95%.

**Verified result (with mean-pool architecture):**
```
Epoch   0,  Loss: 0.692567
Epoch 200,  Loss: 0.065533
Epoch 400,  Loss: 0.004981
Final loss:  0.003044  (↓ 99.6%)
Bit accuracy: 100.00%  (1024/1024 bits correct)  ✓ PASS
```

### 9.4 `test_neural_vs_b1.py` — Performance gating (requires trained checkpoint)

| Test | What it checks | Threshold |
|------|---------------|-----------|
| `test_n1_beats_b1_at_5db` | N1 BLER < B1 BLER at SNR=5dB, INR=5dB | N1 < B1 (any improvement) |
| `test_n1_bler_decreases_with_snr` | BLER at 3, 5, 7 dB is monotonically decreasing | allows ±0.02 tolerance |

**Note:** These tests are marked `@pytest.mark.slow` and `@requires_checkpoint`. They are skipped if `results/phase2a/checkpoints/best_gru.pt` does not exist.

### 9.5 Running tests

```bash
# Smoke tests (fast, no checkpoint needed)
pytest tests/test_neural_ops.py -v
pytest tests/test_neural_overfit.py -v

# Standalone overfit sanity check
python tests/test_overfit_gru_batch.py

# Performance gate (requires trained checkpoint)
pytest tests/test_neural_vs_b1.py -v -m slow
```

---

## 10. Experimental Results

### 10.1 Summary of Experiments

| Experiment | Pooling | SNR (dB) | INR (dB) | Epochs | Best bit_acc | BLER | Notes |
|-----------|---------|----------|----------|--------|-------------|------|-------|
| Exp 1 | Mean | 0–10 (train range) | 0 (fixed) | 200 | ~53.1% | 1.0 | Original config; val at mismatched SNR=5/INR=5 initially |
| Exp 2 | Mean | 15–20 (easy) | −5 to −2.5 | 200 | 53.1% | 1.0 | Curriculum Stage 1; validation matched to training |
| Exp 3 | Mean | 4–10 (curriculum Stage 2) | −5 to 7.5 | ~20 | ~50.3% | 1.0 | Resumed from Exp 2; model had not learned enough to transfer |
| Exp 4 | Attention | 15–20 (easy) | −5 to −2.5 | 200 | 53.8% | 1.0 | Attention pooling; marginal improvement over mean |

### 10.2 Detailed Training Logs

**Experiment 2 — Mean pooling, SNR 15–20 dB, 200 epochs (definitive run):**

```
Epoch   0 | loss=0.6932 | bit_acc=0.5011 | val_bit_acc=0.5010
Epoch  50 | loss=0.6673 | bit_acc=0.5265 | val_bit_acc=0.5267   (LR drop to 1e-4)
Epoch 100 | loss=0.6650 | bit_acc=0.5296 | val_bit_acc=0.5298   (LR drop to 1e-5)
Epoch 150 | loss=0.6643 | bit_acc=0.5305 | val_bit_acc=0.5308   (LR drop to 1e-6)
Epoch 195 | loss=0.6643 | bit_acc=0.5308 | val_bit_acc=0.5309
Final best: val_bit_acc = 0.5310 (epoch 180)
```

The model plateaued around epoch 50–60 and made negligible progress after that despite three learning rate reductions. The loss converged to 0.6643, well above the theoretical minimum.

**Experiment 4 — Attention pooling, SNR 15–20 dB, 200 epochs:**

```
Epoch   0 | loss=0.6932 | bit_acc=0.5011 | val_bit_acc=0.5014
Epoch  50 | loss=0.6649 | bit_acc=0.5313 | val_bit_acc=0.5313
Epoch 100 | loss=0.6656 | bit_acc=0.5269 | val_bit_acc=0.5266   (regression)
Epoch 150 | loss=0.6567 | bit_acc=0.5355 | val_bit_acc=0.5355
Epoch 190 | loss=0.6556 | bit_acc=0.5368 | val_bit_acc=0.5372
Final best: val_bit_acc = 0.5384 (epoch 198)
```

Attention pooling showed slightly more learning capacity (53.8% vs 53.1%) but also exhibited instability — bit accuracy regressed around epoch 60–100 before recovering. The improvement over mean pooling is marginal and both remain far from useful decoding performance.

### 10.3 Analysis

**Initial loss of 0.6932 ≈ ln(2) = 0.6931.** This is the loss of a binary classifier that outputs 0.5 for every bit. The model starts at random guessing and barely moves from there.

**The bottleneck is the architecture, not the training method.** Evidence:
1. Mean pooling at 20 dB SNR: 53.1%. The signal is nearly clean, yet the model cannot learn.
2. Attention pooling at 20 dB SNR: 53.8%. A better pooling method barely helps.
3. Curriculum training from easy to hard: Stage 2 could not build on Stage 1 because Stage 1 had not learned enough.
4. The overfit test passes (100% accuracy on 4 blocks). The model has enough capacity to memorize a tiny batch, but cannot generalize to the full decoding problem.

**Why the architecture fails:** The model must compress 524 received symbols into a 64-dimensional vector (via pooling or hidden states) and then expand that to 256 independent bit decisions. This is a massive information bottleneck. A rate 1/2 K=7 convolutional code has complex dependencies across the entire codeword — the Viterbi algorithm exploits these dependencies through a 64-state trellis search over 262 time steps. The GRU has no mechanism to perform this structured search; it must learn it implicitly from data, which requires far more capacity than h=32 provides.

### 10.4 Debugging Journey

The following issues were identified and fixed during the debugging process:

1. **Critical architecture bug (fixed before experiments above):** The original `forward()` method selected every-other timestep from the GRU's per-timestep output (positions 0, 2, 4, ..., 510) and treated them as info bit predictions. This assumed a correspondence between timestep index and bit index that the GRU had no way to learn. Fixed by switching to mean-pool → Linear(64, 256).

2. **Validation SNR/INR mismatch (fixed):** Validation was hardcoded to SNR=5 dB / INR=5 dB while training at different conditions. BLER was stuck at 1.0, giving no training signal. Fixed by matching validation to training distribution midpoint.

3. **Checkpointing on BLER (fixed):** BLER stayed at 1.0 for the entire training run, so the first epoch was always the "best" checkpoint. Fixed by checkpointing on validation bit accuracy.

---

## 11. Output Artifacts

All saved to `results/phase2a/`:

```
results/phase2a/
├── checkpoints/
│   └── best_gru.pt      # Best GRU model (highest val bit_acc)
├── figures/
│   └── training_loss.pdf         # BCE loss + val bit_acc vs epoch
└── logs/
    └── training_gru_h32.json     # Per-epoch: loss, bit_acc, val_bit_acc, val_bler, lr, time_s
```

> **Note:** LSTM and vanilla RNN variants were not trained, as the GRU (the most capable of the three for sequence tasks) already demonstrated that the end-to-end approach is fundamentally limited at this architecture scale. BLER vs SNR/INR sweep plots were not generated because the model never achieved meaningful decoding performance.

---

## 12. Success Criteria

From CLAUDE.md:

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| N1 vs B1 at BLER = 10⁻³ | N1 must beat B1 by ≥ 0.5 dB | BLER = 1.0 at all conditions | **NOT MET** |
| Training stability | Val BLER decreases monotonically | BLER stuck at 1.0; bit_acc improved but plateaued at ~53% | **NOT MET** |
| Compute budget | Final architecture ops ≤ 2.1M per block | h=32 gives ~6.6M ops (3× over budget) | **NOT MET** |
| Overfit test | Model can memorize 4 training blocks | loss 0.003, 100% accuracy | **MET** |

**Conclusion:** The only criterion met was the overfit sanity check, which confirms the model has enough capacity to memorize a tiny batch. All performance criteria were not met. The end-to-end BiGRU approach is not viable for this decoding task at this architecture scale.

---

## 13. Phase 2a Conclusion and Motivation for Phase 2b

### 13.1 Why End-to-End Decoding Failed

The BiGRU end-to-end decoder failed because the task requires the model to simultaneously solve two problems that are fundamentally different in nature:

1. **Local estimation:** Extracting useful information from each noisy received symbol, accounting for AWGN and sinusoidal interference with unknown parameters.

2. **Global search:** Finding the most likely path through a 64-state trellis over 262 time steps, respecting the constraint structure of the convolutional code.

The Viterbi algorithm solves problem (2) exactly and efficiently using dynamic programming. The BiGRU was asked to learn both problems implicitly from data. With h=32 (64 dims after bidirectional concatenation), the model does not have enough representational capacity to learn the trellis structure. The gradient signal is too diffuse — when a single bit prediction is wrong, backpropagation must determine whether the error was in the GRU's sequence processing (problem 1) or in the output mapping (problem 2), across 524 timesteps and through a mean-pool bottleneck.

### 13.2 Motivation for Phase 2b

Phase 2b uses the **same BiGRU architecture** but repurposes it for **branch metric estimation** instead of end-to-end decoding. The division of labor becomes:

- **BiGRU:** Estimates branch metrics at each trellis section, producing per-timestep outputs that account for interference and noise. This is a local estimation task (problem 1 only).
- **Viterbi algorithm:** Takes the GRU's branch metrics and performs the global trellis search (problem 2).

This decomposition plays to each component's strengths. The GRU handles the part that is hard to model analytically (unknown interference parameters). The Viterbi algorithm handles the part that is well-understood and solved exactly (trellis search).

The Phase 2a negative result makes this comparison clean and compelling: same architecture, same parameter count, different role — expected dramatic improvement.

---

## 15. Key References

| Paper | Relevance |
|-------|-----------|
| Gruber et al., "On Deep Learning-Based Channel Decoding" (2017) | Foundational RNN decoder for convolutional codes |
| Jiang et al., "LEARN Codes: Inventing Low-Latency Codes via RNNs" (2019, IEEE TCOM) | GRU encoder/decoder co-design, robustness under channel mismatch |
| Kim et al., "Communication Algorithms via Deep Learning" (2018) | Bi-GRU decoder with BatchNorm, reference implementation |
| IEEE ICC 2024, "Deep Learning Based Decoder for Concatenated Coding over Deletion Channels" | Bi-GRU as LLR estimator, single network across channel params |

---

## 16. Implementation Checklist

### Completed ✅

- [x] `BiRNNDecoder(nn.Module)` with configurable `hidden_size`, `cell_type`, `bidirectional`, `use_batchnorm`, `input_dim`
- [x] `TrainConfig` dataclass with all hyperparameters and automatic path resolution
- [x] `build_codeword_pool(trellis, pool_size, seed)` — pre-encodes codewords, handles variable-length symbols
- [x] `generate_training_batch(pool, batch_size, snr_range, inr_range, period_range, rng, device)` — fresh channel per block
- [x] `validate_bler(model, trellis, pool, snr_db, inr_db, n_blocks, device, seed)` — BLER + bit accuracy evaluation without gradient
- [x] `train_model(config, seed, model=None)` — full training loop with Adam + StepLR + gradient clipping + early stopping + checkpointing on bit accuracy + JSON logging + optional pretrained model for curriculum training
- [x] `make_decoder_n1(model, device)` — `eval.py`-compatible decode wrapper
- [x] `load_model(checkpoint_path, device)` — reconstruct from checkpoint
- [x] Smoke test in `__main__` block
- [x] `test_neural_ops.py` — architecture and op count tests
- [x] `test_neural_overfit.py` — training sanity check (pytest)
- [x] `test_overfit_gru_batch.py` — standalone overfit script (verified passing: loss 0.003, 100% accuracy)
- [x] `test_neural_vs_b1.py` — performance gating (skips without checkpoint)
- [x] Fixed critical architecture bug: per-timestep output → mean-pool + Linear(64, 256)
- [x] Fixed validation mismatch: val SNR/INR now matches training distribution midpoint
- [x] Fixed checkpointing metric: bit accuracy instead of BLER
- [x] Implemented attention pooling variant
- [x] Implemented curriculum training support (multi-stage with pretrained model passing)
- [x] Trained GRU with mean pooling at SNR 15–20 dB, 200 epochs → 53.1% bit accuracy
- [x] Trained GRU with attention pooling at SNR 15–20 dB, 200 epochs → 53.8% bit accuracy
- [x] Documented negative result and Phase 2a conclusion

### Not Pursued (justified by negative result) ⬜

- [ ] ~~Train LSTM variant~~ — GRU already failed; LSTM unlikely to succeed at matched compute
- [ ] ~~Train vanilla RNN variant~~ — strictly weaker than GRU
- [ ] ~~BLER vs SNR sweep evaluation~~ — model never achieved meaningful decoding
- [ ] ~~BLER vs INR sweep evaluation~~ — same reason
- [ ] ~~Generalization tests~~ — no useful model to test
- [ ] ~~Compare N1 vs B1, B2, B5~~ — N1 BLER = 1.0 at all conditions
- [ ] ~~Generate phase2a_bler_vs_snr.pdf, phase2a_bler_vs_inr.pdf~~ — no meaningful data to plot
- [ ] ~~Find budget-compliant hidden size and retrain~~ — reducing h would only make performance worse

---

*Last updated: April 7, 2026 — Training experiments complete. Negative result documented. Mean pooling (53.1%) and attention pooling (53.8%) both failed. Phase 2a concluded; motivation for Phase 2b established.*