# CLAUDE.md — Search-Designed Trellis Codes with Neural Decoding
**Course:** EE597 | **Author:** Abhishek Manjunath  
**Deliverable:** Simulation results + IEEE-style paper writeup  
**Stack:** Python 3.10+, PyTorch 2.0+, NumPy, SciPy, Matplotlib/Seaborn  
**Compute:** Local GPU/CPU initially; university cluster later (write cluster-ready code from the start)

---

## Project Thesis

> "When channels are mismatched, structured, or computationally constrained, the best end-to-end system may be obtained by co-designing the trellis and a constrained learned decoder for the realistic impairment family."

The goal is **not** to beat polar codes on clean AWGN. The goal is robustness, graceful degradation, and compute efficiency under periodic sinusoidal interference — an industrial IoT-relevant channel.

---

## System Parameters (Fixed — Do Not Change Without Explicit Instruction)

| Parameter | Value |
|-----------|-------|
| Modulation | BPSK ∈ {-1, +1} |
| Info bits | K = 256 |
| Constraint length | K_c = 7 (memory m = 6, so S = 2^m = 64 states) |
| Tail bits | m = 6 zero bits appended to force trellis back to state 0 |
| Total input bits | K + m = 256 + 6 = 262 |
| Coded block length | N = 2 × (K + m) = 2 × 262 = 524 coded bits (includes 12 termination coded bits) |
| Nominal code rate | R = 1/2 (per trellis section) |
| Effective code rate | R_eff = K / N = 256 / 524 ≈ 0.4885 (due to termination overhead) |
| Trellis states | S = 64 |
| Trellis structure | Time-invariant, terminated (forced back to state 0) |
| SNR range | 0–10 dB (focus point: 5 dB) |
| INR range | -5 to 15 dB |
| Interference model | i[t] = A·sin(2πt/P + φ), P ∈ [8,32], φ ∈ [0,2π], A = √(2·INR_linear·σ²) |
| Target BLER eval point | BLER = 10⁻³ |
| Monte Carlo trials | 100,000 (evaluation); 10,000 (search full eval); 1,000 (search proxy) |
| Op budget | 2.1M operations (all methods must match this) |

---

## Codebase Structure

```
project/
├── CLAUDE.md
├── Phase2a_Documentation.md  # [Phase 2a] Full documentation incl. negative result & motivation for 2b
├── channel.py          # AWGN + sinusoidal interference simulator
├── trellis.py          # Trellis FSM: encode, validate, load standard codes
├── decoders.py         # Viterbi (mismatched + oracle), optional C speedup
├── viterbi_core.c      # C extension for Viterbi inner loop
├── viterbi_core.so     # Compiled C extension
├── interference_est.py # FFT-based interference estimation + cancellation
├── baselines.py        # B1, B2, B5 baseline wrappers
├── eval.py             # Monte Carlo BLER, SNR/INR sweeps, Phase 1 runner
├── compute_cost.py     # FLOPs/latency profiler (applies to ALL methods)
├── plot_utils.py       # IEEE-style figures, Paul Tol palette, BLER curves
├── neural_decoder.py   # [Phase 2a] BiRNNDecoder (BiGRU, final-hidden-state pooling), training loop, decode fn
├── neural_bm.py        # [Phase 2b] Neural branch metric estimator + Viterbi integration
├── search.py           # [Phase 3] Evolutionary/SA trellis search — not yet created
├── fitness.py          # [Phase 3] Two-tier fitness: cheap proxy + full eval — not yet created
├── tests/
│   ├── conftest.py             # Pytest configuration
│   ├── test_encode_decode.py   # Noiseless encode→decode round-trip
│   ├── test_awgn_theory.py     # Validate AWGN-only perf vs theory
│   ├── test_oracle_vs_mismatch.py  # Oracle always ≥ mismatched
│   ├── test_bler_vs_snr_plot.py    # 5-curve BLER comparison (B1/B2, AWGN/interference, theory)
│   ├── test_neural_ops.py      # [Phase 2a] Verify GRU/LSTM/RNN op count ≤ 2.1M
│   ├── test_neural_overfit.py  # [Phase 2a] Overfit on 1 batch to verify learning works
│   ├── test_neural_vs_b1.py    # [Phase 2a] N1 must beat B1 by ≥0.5 dB
│   ├── test_overfit_gru_batch.py   # [Phase 2a] Single-batch GRU overfit sanity check
│   ├── test_nbm_ops.py         # [Phase 2b] Verify neural BM + Viterbi total ops ≤ 2.1M
│   ├── test_nbm_overfit.py     # [Phase 2b] Overfit on 1 batch — branch metrics should converge to oracle
│   ├── test_nbm_awgn.py        # [Phase 2b] On pure AWGN, N2 ≈ B1 (no interference to learn)
│   └── test_nbm_vs_b1.py       # [Phase 2b] N2 must beat B1 by ≥0.5 dB under interference
└── results/
    ├── phase1/         # Phase 1 eval: .npz data, compute_table.md
    │   └── figures/    # phase1_bler_vs_snr.pdf, phase1_bler_vs_inr.pdf
    ├── phase2a/        # Phase 2a eval: model checkpoints (empty — training failed), logs
    │   ├── checkpoints/  # Empty — no successful model saved
    │   └── logs/         # Training logs (if any runs completed)
    ├── phase2b/        # Phase 2b eval
    │   ├── checkpoints/  # Trained neural BM models
    │   ├── logs/         # Training logs
    │   └── figures/      # BLER curves, metric visualizations
    └── test/           # Test-generated plots
```

---

## Current Repo Snapshot (2026-04-08)

**Git state:** clean working tree.

**What is present now:**
- Phase 1 baseline code, figures, `.npz` sweep outputs, and `results/phase1/compute_table.md` are checked in.
- Phase 2a code is present in `neural_decoder.py`; `results/phase2a/checkpoints/` and `results/phase2a/logs/` exist but are empty.
- Phase 2b code is present in `neural_bm.py` and the associated tests exist under `tests/`.
- One Phase 2b checkpoint is checked in: `results/phase2b/checkpoints/best_model_seed99.pt`.
- No Phase 2b logs or figures are currently checked in.

**Important repo-level status notes:**
- The Phase 2b tests `tests/test_nbm_vs_b1.py` currently look for `best_model_seed42.pt`, but the committed checkpoint is `best_model_seed99.pt`.
- `search.py` and `fitness.py` are still not present; trellis search has not started.
- `pytest` was not available on the current shell path during this documentation update, so the status below is based on repository contents rather than a fresh full test run.

---

## Approaches to Implement (All Phases)

### Baselines (Phase 1)
| ID | Name | Description |
|----|------|-------------|
| B1 | Mismatched Viterbi | NASA K=7 conv code (d_free=10), decoded with AWGN-only metrics — ignores interference |
| B2 | Oracle Viterbi | Same code, Viterbi with perfect interference knowledge — upper bound |
| B3 | Random Trellis | Random 64-state trellis with same neural decoder — sanity check for search |
| B4 | Polar SCL | CRC-aided polar code, small fixed list size, mismatched decoder |
| B5 | Interference Cancel + Viterbi | **(Professor addition)** Estimate & subtract interference, then standard Viterbi |

### Neural Approaches (Phase 2)
| ID | Name | Description |
|----|------|-------------|
| N1 | GRU End-to-End | Bidirectional GRU (h=32/dir), ~6.6K params, trained on distribution of channel params. Also test LSTM (h=26/dir) and vanilla RNN as comparisons. |
| N2 | Neural Branch Metric | **(Professor addition)** Viterbi/BCJR structure kept intact; NN replaces branch metric computation — structured hybrid |

### Proposed System (Phases 3–4)
| ID | Name | Description |
|----|------|-------------|
| S1 | Searched Trellis + GRU | Core contribution: evolutionary search finds best 64-state trellis, decoded by N1 |
| S2 | Searched Trellis + Neural BM | Optional: searched trellis decoded by N2 |

---

## Compute Cost — CRITICAL REQUIREMENT (Professor Feedback)

**Every approach must have its compute cost measured and reported. No comparison is valid without this.**

- Use `compute_cost.py` to profile every method: FLOPs, wall-clock time per block, memory
- The constraint is 2.1M operations — enforce this across ALL methods being compared
- For neural methods: use `thop` or `pytorch-OpCounter` to count ops
- For classical methods: count multiply-adds analytically (Viterbi: O(S²·N), where N=524)
- Report as a table in the paper: method vs BLER vs ops vs latency

```python
# compute_cost.py must expose:
def count_ops(model_or_fn, sample_input) -> int: ...
def measure_latency(model_or_fn, sample_input, n_runs=1000) -> float: ...  # ms per block
```

---

## Neural Branch Metric Estimator — Full Design Specification (Phase 2b)

### Concept

Keep the Viterbi dynamic programming structure intact. Replace *only* the branch metric computation γ(s→s', y_t) with a small neural network. The NN learns to produce interference-aware log-likelihood scores; Viterbi handles the global trellis search optimally given those scores.

### Why This Works (and Phase 2a Didn't)

Phase 2a failed because a ~23K-parameter BiGRU was asked to implicitly learn the Viterbi search over 64 states × 524 time steps — a combinatorially hard sequence-to-vector mapping. Phase 2b decomposes the problem:

| | Phase 2a (failed) | Phase 2b (proposed) |
|---|---|---|
| **Task** | Predict 256 info bits from 524 received samples | Predict branch metrics at each of 524 time steps |
| **Output shape** | (batch, 256) — sequence-to-vector | (batch, 262, num_branch_outputs) — sequence-to-sequence |
| **What NN learns** | Implicit Viterbi search (impossible at this scale) | Local channel likelihood estimation (simple) |
| **Who does the search** | The NN alone | Viterbi algorithm (optimal DP) |
| **NN complexity needed** | Very large (would need to represent 64-state trellis) | Small (only needs to track a 3-parameter sinusoid) |
| **Training target** | Info bits via BCE | Oracle branch metrics via MSE |

### Architecture: BiGRU Sequence-to-Sequence Branch Metric Estimator

```
Input:  (batch, T, 1)              — T = K + m = 262 trellis steps, each with
                                      rate-1/2 outputs, so 2 received samples per step
        Reshape to (batch, T, 2)   — pair received samples per trellis section

BiGRU:  hidden_size=h per direction, num_layers=1
        Input:  (batch, T, 2)
        Output: (batch, T, 2h) — all hidden states, both directions

Linear: Linear(2h, num_branch_outputs) per time step
        Output: (batch, T, num_branch_outputs) — one score per possible branch output

Interpretation: output[b, t, j] = learned branch metric for branch output j at time t
                These replace γ(s→s', y_t) in Viterbi ACS
```

**Key design choice — input grouping:** The trellis has 262 sections (K+m=262), each producing 2 coded bits. Group received samples as pairs (y_{2t}, y_{2t+1}) per trellis step, giving input shape (batch, 262, 2). This aligns the NN's time axis with the trellis time axis.

**Number of branch outputs:** For rate-1/2 code, each trellis section has 2 possible input bits (0 or 1), producing one of 2 possible output pairs. But multiple states can share the same output pair. The NN outputs a score for each of the 4 possible BPSK output pairs: {(-1,-1), (-1,+1), (+1,-1), (+1,+1)}. During Viterbi, for each transition (s→s'), look up which output pair that transition produces and use the corresponding NN score.

So `num_branch_outputs = 4`.

### Hidden Size Selection and Compute Budget

The NN must estimate A·sin(2πt/P + φ) — a sinusoid with 3 continuous parameters (A, P, φ) that vary per block. The BiGRU hidden state needs enough capacity to infer these from noisy observations. Start with h=16 per direction:

| Component | Op Count (approx) |
|-----------|-------------------|
| BiGRU (h=16, T=262, input=2) | 262 × 2(dirs) × 3(gates) × 16 × (16+2) × 2(mul+add) ≈ 905K |
| Linear (32→4, T=262) | 262 × 32 × 4 ≈ 34K |
| Viterbi ACS (64 states, 262 steps, 2 branches/state) | 262 × 64 × 2 × ~4 ops ≈ 134K |
| **Total** | **≈ 1.07M** |

Well under 2.1M budget. This leaves room to increase h if needed.

**Hyperparameter sweep plan:** Test h ∈ {8, 12, 16, 24}. Expect diminishing returns above h=16 for this task complexity.

### Training Procedure

**Training targets — Oracle branch metrics:**

For each training sample, we know the true interference parameters (A, P, φ) because we generate the data. Compute oracle branch metrics:

```python
# For trellis step t, branch output pair (c0, c1) mapped to BPSK (x0, x1):
# Oracle metric = -[(y_{2t} - x0 - interference_{2t})² + (y_{2t+1} - x1 - interference_{2t+1})²] / (2σ²)
# This is the log-likelihood under the true channel model.
```

The NN is trained to match these oracle metrics via MSE loss.

**Why MSE on oracle metrics (not BCE on bits):** The branch metric is a continuous scalar representing log-likelihood. MSE directly measures how well the NN approximates the oracle metric. There is no classification boundary to learn — just a regression target. This is also what ViterbiNet (Shlezinger et al.) and BCJRNet use.

**Training configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Loss | MSE on branch metrics | Regression target, not classification |
| Optimizer | Adam | Standard for small NNs |
| Learning rate | 1e-3, decay by 0.5 every 50 epochs | Staircase decay (per Kargı et al.) |
| Batch size | 256 | On-the-fly generation, no dataset size limit |
| Training SNR | Uniform over [1, 7] dB | Covers focus range; avoid extremes |
| Training INR | Uniform over [-5, 15] dB | Full INR range |
| Training P | Uniform integer over [8, 32] | Full period range |
| Training φ | Uniform over [0, 2π] | Full phase range |
| Epochs | 200 (with early stopping, patience=20) | |
| Data generation | On-the-fly per mini-batch | No static dataset; infinite variety |
| Gradient clipping | max_norm=1.0 | Stabilize BiGRU training |
| Batch normalization | Between BiGRU output and linear head | Per Kargı et al. and Zhang & Luo |

**Validation:** Every 5 epochs, run 1,000 blocks at SNR=5 dB, INR=5 dB with Viterbi decoding using the NN's branch metrics. Track BLER as the primary validation metric (not just MSE).

### Viterbi Integration

The existing `decoders.py` Viterbi implementation computes branch metrics analytically:

```python
# Current (B1 mismatched): γ = -||y - x_hypothesis||² / (2σ²)
# Current (B2 oracle):     γ = -||y - x_hypothesis - interference||² / (2σ²)
```

For N2, replace this with:

```python
# N2 (neural BM): γ(s→s', t) = nn_output[t, branch_output_index(s, s')]
```

**Implementation approach:** Create a new Viterbi function `viterbi_neural_bm(nn_branch_metrics, trellis)` that:
1. Takes pre-computed NN branch metrics of shape (T, 4) — already computed in a single forward pass
2. For each trellis step t and each transition (s→s'), looks up the NN score for the output pair that transition produces
3. Runs standard ACS (add-compare-select) using these scores
4. Returns decoded bits via traceback

This means the NN forward pass happens ONCE for the whole block, producing all branch metrics. Then Viterbi runs using those metrics. No per-step NN calls.

### Sanity Checks and Tests (implement in order)

1. **test_nbm_ops.py:** Verify total ops (NN forward + Viterbi) ≤ 2.1M
2. **test_nbm_overfit.py:** Fix one (SNR, INR, P, φ) setting. Train on repeated identical batches. NN branch metrics should converge toward oracle metrics (MSE → ~0). Viterbi with these metrics should achieve ~0 BLER at high SNR.
3. **test_nbm_awgn.py:** On pure AWGN (INR=-∞), N2 should perform ≈ B1. The NN should learn that branch metrics are just standard Euclidean distance. This validates the integration doesn't break anything.
4. **test_nbm_vs_b1.py:** Under interference (INR=5 dB), N2 should beat B1 by ≥0.5 dB at BLER=10⁻³.

### Implementation Steps (Ordered)

**Step 1 — `neural_bm.py` core module:**
- [ ] `NeuralBranchMetric(nn.Module)`: BiGRU + BN + Linear, configurable h
- [ ] `compute_oracle_metrics(y, x_bpsk, interference, sigma)`: ground truth targets
- [ ] `viterbi_neural_bm(branch_metrics, trellis)`: Viterbi using NN metrics, returns decoded bits
- [ ] `__main__` smoke test: random input → forward pass → shapes correct → Viterbi runs

**Step 2 — Training loop:**
- [ ] `train_neural_bm(config, seed)`: on-the-fly batch generation, MSE loss on oracle metrics, BLER-based validation
- [ ] Checkpoint saving/loading
- [ ] Training log output (epoch, MSE, val_BLER)

**Step 3 — Tests (in order):**
- [ ] `test_nbm_ops.py` — compute budget
- [ ] `test_nbm_overfit.py` — learning works
- [ ] `test_nbm_awgn.py` — doesn't break clean channel
- [ ] `test_nbm_vs_b1.py` — beats mismatched Viterbi

**Step 4 — Evaluation:**
- [ ] `make_decoder_n2(model, device)`: inference wrapper compatible with `eval.py`
- [ ] BLER vs SNR sweep (N2, B1, B2, B5) at INR=5 dB
- [ ] BLER vs INR sweep (N2, B1, B2, B5) at SNR=5 dB
- [ ] Compute cost table for N2
- [ ] Distribution shift test: train on P∈[8,32], test on P∈[4,8]∪[32,48]

**Step 5 — Ablations (if N2 works):**
- [ ] Hidden size sweep: h ∈ {8, 12, 16, 24}
- [ ] Context window: compare BiGRU (full context) vs MLP (local window only) vs 1D-CNN
- [ ] Visualize learned metrics vs oracle metrics at sample time steps

### Key References for Phase 2b

- Shlezinger et al., "ViterbiNet: A Deep Learning Based Viterbi Algorithm for Symbol Detection" (IEEE TWC, 2020) — DNN replaces branch metric in Viterbi; hybrid model-based/data-driven
- Yang & Jiang, "Online Learning of Trellis Diagram Using Neural Network" (arXiv:2202.10635) — learns trellis diagram via ANN + pilot; integrates with Viterbi and BCJR
- NN-Aided BCJR (Wang et al., IEEE VTC 2020) — NN replaces channel-model-based branch probability in BCJR; 2.3 dB gain over separate detection+decoding
- Kargı & Duman, "A Deep Learning Based Decoder for Concatenated Coding over Deletion Channels" (ICC 2024) — BI-GRU as LLR estimator, 4–8 layers, hidden 128–1024, MSE/BCE loss, staircase LR decay, batch norm between layers
- Yan et al., "Decoding for Punctured Convolutional and Turbo Codes" (arXiv:2502.15475, Feb 2025) — LSTM/BiGRU neural decoders, puncturing-aware embedding, generalizes across block lengths
- Streit et al., "BCJRFormer" (ISIT 2025) — Transformer replaces BCJR for marker codes; relevant as next-generation alternative to BiGRU approach

---

## Interference Estimation + Cancellation Baseline (B5, Phase 1/2)

**Concept:** Estimate the sinusoidal interference parameters (A, P, φ) from the received signal, subtract it, then run standard Viterbi on the cleaned signal.

**Implementation approach:**
1. Use periodogram / FFT to detect dominant frequency in r[t] - x̂[t] (or from pilots)
2. Fit amplitude and phase via least squares
3. Subtract estimated interference: ỹ[t] = r[t] - Â·sin(2πt/P̂ + φ̂)
4. Run mismatched Viterbi on ỹ[t]

**Why this is a fair and important baseline:** If simple DSP cancellation already solves the problem, the neural approach has no value. We need to show the regime where estimation error limits this approach.

---

## Phase Milestones

### Phase 0 — Specification ✅ COMPLETE
- [x] System parameters defined
- [x] Channel model specified
- [x] Baselines identified
- [x] Success criteria defined

---

### Phase 1 — Strong Baselines ✅ COMPLETE
**Goal:** Reproduce reliable BLER curves for all classical baselines. Validate simulation framework.

**Tasks:**
- [x] `channel.py`: AWGN + sinusoidal interference, verified A formula
- [x] `trellis.py`: FSM encode, NASA K=7 code, non-catastrophic + connectivity validation
- [x] `decoders.py`: Viterbi with mismatched AWGN metric (B1)
- [x] `decoders.py`: Viterbi with oracle metric — knows A, P, φ (B2)
- [x] `baselines.py`: interference cancellation + Viterbi (B5)
- [x] `compute_cost.py`: FLOPs and latency profiling
- [x] `eval.py`: Monte Carlo BLER loop, SNR sweep + INR sweep, plotting
- [x] **Validation tests** (all passing):
  - Noiseless encode→decode = 0 errors
  - AWGN-only BLER matches theoretical K=7 performance
  - Oracle Viterbi always ≥ Mismatched Viterbi (no exceptions)

**Deliverables (in `results/phase1/`):**
- BLER vs SNR curves for B1, B2, B5 + uncoded BPSK theory at INR=5 dB
- BLER vs INR curves for B1, B2, B5 at SNR=5 dB (INR=-5 to 15 dB)
- Compute cost table (all methods within 2.1M op budget)

---

### Phase 2 — Neural Decoder Baselines
**Goal:** Validate that learned decoding improves robustness under sinusoidal interference.

**Sub-phases:**
#### 2a — GRU End-to-End Decoder (N1) ✅ COMPLETE — **Negative Result**

**Outcome:** The BiGRU end-to-end decoder cannot learn to decode the rate-1/2 K=7 convolutional code. After extensive experimentation (mean pooling, final-hidden-state pooling, attention pooling, curriculum training, high-SNR training at 15–20 dB), the model plateaued at ~53% bit accuracy. Random guessing is 50%. BLER remained at 1.0 throughout. None of the success criteria were met.

**Full documentation:** `Phase2a_Documentation.md`

**Architecture implemented (`neural_decoder.py`):**

| Layer | Detail |
|-------|--------|
| Input | (batch, 524, 1) — raw received signal |
| BiGRU | h=32/dir → all hidden states (batch, 524, 64) |
| Pooling | Final hidden states: h_n.transpose(0,1).reshape(B, 64) |
| Output | Linear(64, 256) → (batch, 256) info bit logits |

**Pooling strategies tried and failed:**

| Strategy | Best bit accuracy | Notes |
|----------|------------------|-------|
| Mean pooling | ~52.1% | Uniform weighting over 524 steps |
| Final hidden states | ~53.5% | Current active version in `neural_decoder.py` |
| Attention pooling | ~53.8% | Learnable per-timestep weights; marginal gain |

All variants: BLER = 1.0 (no block ever decoded correctly).

**Conclusion:** End-to-end decoding of a convolutional code requires the model to learn an implicit Viterbi search over 64 states × 524 time steps — a combinatorially hard mapping that a ~23K-parameter BiGRU cannot represent. The architecture is insufficient for this task. See `Phase2a_Documentation.md` §13 for full analysis.

**Implementation tasks completed:**
- [x] `neural_decoder.py`: `BiRNNDecoder(nn.Module)` with configurable hidden_size, cell_type, bidirectional, use_batchnorm
- [x] `neural_decoder.py`: `train_model(config, seed, model)` with on-the-fly batch generation, early stopping, checkpointing
- [x] `neural_decoder.py`: `make_decoder_n1(model, device)` inference wrapper compatible with eval.py
- [x] `neural_decoder.py`: `load_model(checkpoint_path)`, `build_codeword_pool()`, `validate_bler()`
- [x] Tests: `test_neural_ops.py`, `test_neural_overfit.py`, `test_neural_vs_b1.py`, `test_overfit_gru_batch.py`
- [x] Three pooling strategies implemented and benchmarked
- [x] Curriculum training attempted (high SNR → low SNR)

**Key References**

- Gruber et al., "On Deep Learning-Based Channel Decoding" (2017) — foundational RNN decoder for conv codes
- Jiang et al., "LEARN Codes: Inventing Low-Latency Codes via RNNs" (2019, IEEE TCOM)
- Kim et al., "Communication Algorithms via Deep Learning" (2018) — Bi-GRU decoder with BatchNorm

#### 2b — Neural Branch Metric Estimator (N2) ← **IN PROGRESS**

**Motivation (from Phase 2a negative result):** End-to-end decoding failed because a small BiGRU cannot implicitly learn Viterbi over 64 states. Solution: keep the Viterbi/BCJR graph structure intact and replace only the branch metric computation with a small neural network. The NN needs to learn only a local channel likelihood — a much simpler mapping than end-to-end decoding.

**Full architecture specification:** See "Neural Branch Metric Estimator — Full Design Specification" section above.

**Current implementation status:**

- [x] **Step 1 — `neural_bm.py` core module implemented**
  - [x] `NeuralBranchMetric(nn.Module)`: BiGRU + BatchNorm + Linear head
  - [x] `compute_oracle_metrics(...)`
  - [x] `viterbi_neural_bm(branch_metrics, trellis, index_table)`
  - [x] `make_decoder_n2(...)` wrapper for `eval.py`
  - [x] `load_model(...)`
  - [x] `__main__` smoke test / CLI entry point with optional `--train`
- [x] **Step 2 — Training loop implemented**
  - [x] On-the-fly batch generation
  - [x] MSE loss on oracle metrics
  - [x] BLER-based validation
  - [x] Early stopping and checkpoint save/load
  - [ ] Persisted training logs are not currently checked in
- [x] **Step 3 — Test files implemented**
  - [x] `test_nbm_ops.py`
  - [x] `test_nbm_overfit.py`
  - [x] `test_nbm_awgn.py`
  - [x] `test_nbm_vs_b1.py`
  - [ ] Current repo status does not confirm a fresh passing run for the full Phase 2b suite
  - [ ] `test_nbm_vs_b1.py` expects `best_model_seed42.pt`, but the committed checkpoint is `best_model_seed99.pt`
- [ ] **Step 4 — Evaluation incomplete**
  - [ ] BLER vs SNR curves (N2, B1, B2, B5) not checked in
  - [ ] BLER vs INR curves not checked in
  - [ ] Compute cost table for N2 not checked in
  - [ ] Distribution shift tests not implemented
- [ ] **Step 5 — Ablations incomplete**
  - [ ] Hidden size sweep h ∈ {8, 12, 16, 24}
  - [ ] Visualize learned vs oracle metrics

**Artifacts currently committed for N2:**
- `results/phase2b/checkpoints/best_model_seed99.pt`

**Immediate next cleanup item:**
- Align the Phase 2b checkpoint naming and test assumptions so `test_nbm_vs_b1.py` and any related evaluation scripts target the checkpoint that actually exists in the repo.

**Deliverable:** BLER curves + compute table for N2, B1, B2, B5. Evidence of robustness to distribution shift.

---

### Phase 3 — Trellis Search with Decoder-in-the-Loop (Target: ~4–8 weeks)
**Goal:** Discover a 64-state trellis better suited to sinusoidal interference + neural decoding than the classical K=7 code.

**Tasks:**
- [ ] `search.py`: evolutionary algorithm
  - Population initialization: perturb K=7 code (not pure random)
  - Mutation: random transition table rewiring, output label flips
  - Crossover: combine parent trellises
  - Selection: tournament selection
  - Constraint enforcement: non-catastrophic, fully connected, d_free ≥ 8
- [ ] `fitness.py`: two-tier evaluation
  - **Cheap (every generation):** 1,000 MC trials, single SNR/INR point, no NN retraining
  - **Full (every N generations):** 10,000 trials, multiple SNR values, with NN retraining
- [ ] Run multiple independent seeds to avoid local optima
- [ ] Generate Pareto frontier: BLER vs compute for top candidate trellises

**Deliverable:** Set of candidate trellises; fitness improvement curve over generations; best trellis vs B3 (random) must show ≥1 dB gain.

---

### Phase 4 — Analysis, Ablations, and Paper (Target: ~2–4 weeks)
**Goal:** Understand *why* joint co-design helps, and write it up clearly.

**Ablation matrix (all at equal compute budget):**

| System | Trellis | Decoder | Purpose |
|--------|---------|---------|---------|
| B1 | K=7 classical | Mismatched Viterbi | Primary baseline |
| B2 | K=7 classical | Oracle Viterbi | Upper bound |
| B5 | K=7 classical | IC + Viterbi | DSP cancellation baseline |
| N1 | K=7 classical | GRU end-to-end | Isolates decoder contribution |
| N2 | K=7 classical | Neural BM + Viterbi | Structured hybrid |
| S1 | Searched | GRU end-to-end | **Core result** |
| S2 | Searched | Neural BM | Optional extension |
| B3 | Random | GRU | Validates search is non-trivial |

**Stress tests:**
- [ ] Shift INR by +5 dB beyond training — graceful degradation?
- [ ] Shift P outside [8,32] range — generalization?
- [ ] Test SNR range [0,10] vs [5,15] — distribution shift?

**Interpretability:**
- [ ] Which error events dominate? Analyze error patterns
- [ ] Does searched trellis show implicit interleaving structure?
- [ ] Does neural decoder exploit periodicity of interference?

**Paper writeup checklist:**
- [ ] System model section (channel, trellis, decoder)
- [ ] Baselines section (B1–B5 with compute table)
- [ ] Neural approaches section (N1, N2 with architecture diagrams)
- [ ] Search algorithm section
- [ ] Results: BLER curves, compute-matched comparisons, ablation table
- [ ] Discussion: when does each approach win, and why

---

## Success Criteria

| Phase | Criterion |
|-------|-----------|
| Phase 1 | Validation tests pass; baseline BLER curves match theory |
| Phase 2a | ~~N1 beats B1 by ≥0.5 dB~~ — **negative result**: BiGRU end-to-end cannot decode K=7 code; pivoting to N2 (neural branch metric) |
| Phase 2b | N2 beats B1 by ≥0.5 dB at BLER=10⁻³; N2 shows robustness under distribution shift |
| Phase 3 | Searched trellis beats random (B3) by ≥1 dB; search shows fitness improvement over generations |
| Phase 4 | S1 beats B1 at equal 2.1M op budget; all ablations complete; compute costs reported for every method |

---

## Failure Modes to Avoid
- Neural decoder worse than mismatched Viterbi → core thesis fails
- Searched trellis ≈ random trellis → search not working
- System only works at one SNR/INR point → overfitting
- Any comparison made without matching compute budget → unfair, rejected by reviewers

---

## Coding Conventions
- All functions typed with Python type hints
- No global state
- **Never hardcode N=524 or K=256 as literals.** Derive coded block length as `N = 2 * (K + memory)` where `memory = K_c - 1 = 6`. Use config dicts or function parameters.
- Every module has a `__main__` block with a basic smoke test
- Random seeds: always pass `seed` argument explicitly, never rely on global state
- Cluster-ready: no hardcoded local paths; use `pathlib.Path` and config dicts
- Save all results as `.npz` + `.json` (not just plots) for reproducibility

---

## Future Extensions (Do Not Implement Now)
- Fading on top of AWGN (professor mentioned)
- Structured search over polynomial-generated trellises (professor mentioned)  
- Rate 2/3 generalization
- Quantized LLRs / reduced precision

## Skills
- Simulation code: read `.claude/skills/simulation/SKILL.md` before writing channel/trellis/decoder code
- Evaluation: read `.claude/skills/evaluation/SKILL.md` before writing eval or compute-cost code
- Plotting: read `.claude/skills/plotting/SKILL.md` before writing plotting or figure-generation code
- Documentation: read `.claude/skills/documentation/SKILL.md` before writing documentation when prompted to
