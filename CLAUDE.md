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
├── Phase_3                    # [Phase 3] Research-meeting notes on Phase 3 scope and methodology
├── channel.py          # AWGN + sinusoidal interference simulator
├── trellis.py          # Trellis FSM: encode, validate, load standard codes
├── decoders.py         # Viterbi (mismatched + oracle), optional C speedup
├── viterbi_core.c      # C extension for Viterbi inner loop
├── viterbi_core.so     # Compiled C extension
├── interference_est.py # FFT-based interference estimation + cancellation
├── baselines.py        # B1, B2, B3, B5 baseline wrappers
├── eval.py             # Monte Carlo BLER, SNR/INR sweeps, Phase 1 runner
├── compute_cost.py     # FLOPs/latency profiler (applies to ALL methods)
├── plot_utils.py       # IEEE-style figures, Paul Tol palette, BLER curves
├── plot_training_history.py  # [Phase 2b] Plot training/validation loss curves
├── neural_decoder.py   # [Phase 2a] BiRNNDecoder (BiGRU, final-hidden-state pooling), training loop, decode fn
├── neural_bm.py        # [Phase 2b] Neural branch metric estimator + Viterbi integration
├── trellis_genome.py   # [Phase 3] Candidate representation + perturbation operator — not yet created
├── constraints.py      # [Phase 3] Non-catastrophic, connectivity, d_free — not yet created
├── search.py           # [Phase 3] Evolutionary trellis search (fitness-callback driven) — not yet created
├── fitness.py          # [Phase 3] fitness_oracle (3a) + fitness_n2 (3b) — not yet created
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
│   ├── test_nbm_vs_b1.py       # [Phase 2b] N2 must beat B1 by ≥0.5 dB under interference
│   ├── test_constraints.py     # [Phase 3] d_free(NASA K=7)=10; non-catastrophic + connectivity — not yet created
│   ├── test_genome.py          # [Phase 3] Round-trip serialize/deserialize; perturbation validity rate — not yet created
│   ├── test_n2_generalization.py  # [Phase 3b prerequisite] Frozen N2 on 10 random valid trellises — not yet created
│   └── test_search_convergence.py # [Phase 3] EA fitness curve monotone non-increasing — not yet created
└── results/
    ├── phase1/         # Phase 1 eval: .npz data, compute_table.md
    │   └── figures/    # phase1_bler_vs_snr.pdf, phase1_bler_vs_inr.pdf
    ├── phase2a/        # Phase 2a eval: model checkpoints (empty — training failed), logs
    │   ├── checkpoints/  # Empty — no successful model saved
    │   └── logs/         # Training logs (if any runs completed)
    ├── phase2b/        # Phase 2b eval
    │   ├── checkpoints/  # best_model_seed32.pt, best_model_seed42.pt, best_model_seed99.pt (legacy)
    │   ├── logs/         # history_seed{32,42}.npz
    │   └── figures/      # BLER curves, training history plots
    ├── phase3a/        # [Phase 3a] Oracle-fitness EA results — not yet populated
    ├── phase3b/        # [Phase 3b] N2-fitness EA results — not yet populated
    └── test/           # Test-generated plots
```

---

## Current Repo Snapshot (2026-04-20)

**Git state:** clean working tree.

**What is present now:**
- Phase 1 baseline code, figures, `.npz` sweep outputs, and `results/phase1/compute_table.md` are checked in.
- Phase 2a code is present in `neural_decoder.py`; `results/phase2a/checkpoints/` and `results/phase2a/logs/` exist but are empty (Phase 2a was a negative result).
- Phase 2b code is present in `neural_bm.py` and the associated tests exist under `tests/`.
- Three Phase 2b checkpoints are checked in: `results/phase2b/checkpoints/best_model_seed42.pt`, `best_model_seed32.pt`, and `best_model_seed99.pt` (legacy).
- Phase 2b results are generated and checked in for both seed42 and seed32 models:
  - BLER vs SNR curves (`phase2b_bler_vs_snr_seed{42,32}.{pdf,png}`)
  - BLER vs INR curves (`phase2b_bler_vs_inr_seed{42,32}.{pdf,png}`)
  - Training history plots (`training_history_seed{42,32}.{pdf,png}`)
  - Raw `.npz` data for both sweeps and both seeds
  - Training/validation logs (`history_seed{42,32}.npz`)
  - Compute cost tables (`compute_table_seed32.md`, `compute_table_seed42.md`)
- Phase 3 planning document (`Phase_3`) exists with methodology, search-space definition, fitness-function decisions, and a decomposition into 3a/3b.

**Two model variants (Phase 2b):**

| Seed | Oracle metric target | σ² normalization | BLER @ SNR=5dB, INR=5dB | Description |
|------|---------------------|-----------------|--------------------------|-------------|
| seed42 | −‖y−h−i‖² / (2σ²) | Baked into training target | 2.44e-1 | NN must learn both oracle distance and noise variance scaling |
| seed32 | −‖y−h−i‖² (unnormalized) | Applied post-inference: `bm / (2σ²)` | 2.50e-3 | NN learns only oracle distance; deterministic σ² scaling applied externally at decode time |

The seed32 variant significantly outperforms seed42. It is the current default in `neural_bm.py` — `compute_oracle_metrics()` returns unnormalized distances, and `make_decoder_n2()` divides by `2σ²` after the NN forward pass. **Seed32 is also the default frozen checkpoint for Phase 3b fitness.**

**Important repo-level status notes:**
- `trellis_genome.py`, `constraints.py`, `search.py`, and `fitness.py` are still not present; trellis search has not started. Phase 3 is scoped and planned (see `Phase_3` doc) but no code has been written.

---

## Approaches to Implement (All Phases)

### Baselines (Phase 1)
| ID | Name | Description |
|----|------|-------------|
| B1 | Mismatched Viterbi | NASA K=7 conv code (d_free=10), decoded with AWGN-only metrics — ignores interference |
| B2 | Oracle Viterbi | Same code, Viterbi with perfect interference knowledge — upper bound |
| B3 | Random Trellis + N2 | Random valid 64-state trellis decoded with **N2 (neural BM)** — sanity check that search is non-trivial. Both B3 and S1 must use the same decoder (N2) for the ≥1 dB comparison to be meaningful. |
| B4 | Polar SCL | CRC-aided polar code, small fixed list size, mismatched decoder |
| B5 | Interference Cancel + Viterbi | **(Professor addition)** Estimate & subtract interference, then standard Viterbi |

### Neural Approaches (Phase 2)
| ID | Name | Description |
|----|------|-------------|
| ~~N1~~ | ~~GRU End-to-End~~ | ~~Bidirectional GRU (h=32/dir), ~6.6K params.~~ **Phase 2a negative result: cannot decode K=7 code. Retained in code for reproducibility but out of scope for all Phase 3+ comparisons.** |
| N2 | Neural Branch Metric | **(Professor addition)** Viterbi/BCJR structure kept intact; NN replaces branch metric computation — structured hybrid. **Primary decoder for Phase 3b and S1.** |

### Proposed System (Phases 3–4)
| ID | Name | Description |
|----|------|-------------|
| S1 | Searched Trellis + N2 | **Core contribution (post–Phase 2a pivot):** evolutionary search (Phase 3b) finds best 64-state trellis for the sinusoidal-interference channel when decoded by N2. |
| S1-oracle | Searched Trellis (3a) + Oracle Viterbi | Phase 3a intermediate result. Establishes oracle-decoding ceiling; validates EA machinery. |
| ~~S2~~ | ~~Searched Trellis + N2~~ | ~~Formerly listed as optional; merged into S1 after Phase 2a pivot.~~ |

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

**IMPORTANT for Phase 3b:** The NN output is a function of *received signal only*, producing 4 scores per time step for the 4 possible BPSK output pairs. The trellis structure enters *only* through the output-pair-index table used during Viterbi ACS. This is what makes the same frozen N2 network pluggable into different trellises during the Phase 3b search — provided the generalization sanity check passes.

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
  - [x] Training logs checked in (`results/phase2b/logs/history_seed{42,32}.npz`)
  - [x] Two model variants trained: seed42 (σ²-normalized targets) and seed32 (unnormalized targets, external σ² scaling)
- [x] **Step 3 — Test files implemented**
  - [x] `test_nbm_ops.py`
  - [x] `test_nbm_overfit.py`
  - [x] `test_nbm_awgn.py`
  - [x] `test_nbm_vs_b1.py`
- [x] **Step 4 — Evaluation results generated**
  - [x] BLER vs SNR curves (N2, B1, B2, B5) for both seed42 and seed32
  - [x] BLER vs INR curves for both seed42 and seed32
  - [x] Training history plots for both seeds
  - [x] Raw `.npz` sweep data checked in
  - [x] Compute cost tables checked in (`results/phase2b/compute_table_seed32.md`, `compute_table_seed42.md`)
  - [ ] Distribution shift tests not implemented
- [ ] **Step 5 — Ablations incomplete**
  - [ ] Hidden size sweep h ∈ {8, 12, 16, 24}
  - [ ] Visualize learned vs oracle metrics

**Artifacts currently committed for N2:**
- `results/phase2b/checkpoints/best_model_seed32.pt` — NN learns oracle_distance only (σ² applied externally); **BLER=2.50e-3 @ SNR=5dB, INR=5dB — primary model**
- `results/phase2b/checkpoints/best_model_seed42.pt` — NN learns oracle_distance/σ² (normalized targets); BLER=2.44e-1 @ 5dB — underperforms seed32
- `results/phase2b/checkpoints/best_model_seed99.pt` — legacy checkpoint
- `results/phase2b/figures/` — BLER vs SNR, BLER vs INR, training history plots for both seeds
- `results/phase2b/logs/` — training/validation history for both seeds
- `results/phase2b/*.npz` — raw BLER sweep data for both seeds
- `results/phase2b/compute_table_seed32.md`, `compute_table_seed42.md` — N2 FLOPs=913,664 (43.5% of 2.1M budget)

**Deliverable:** BLER curves + compute table for N2, B1, B2, B5. Evidence of robustness to distribution shift.

---

### Phase 3 — Trellis Search with Decoder-in-the-Loop (Target: ~4–8 weeks)
**Goal:** Discover a 64-state trellis better suited to sinusoidal interference + neural decoding than the classical K=7 code.

**Decomposition rationale.** Phase 3 is split into **3a** (oracle fitness) and **3b** (N2 fitness) to isolate two orthogonal sources of risk:

1. **Is the EA machinery correct?** — genome representation, constraint checks (non-catastrophic, connectivity, d_free), mutation/crossover operators, selection pressure, seed reproducibility.
2. **Does a frozen N2 usefully decode arbitrary valid trellises?** — an unproven assumption; if N2's branch-metric head doesn't generalize across transition tables, Tier 1 fitness collapses.

Running N2-in-the-loop EA directly conflates both risks: a flat fitness curve could come from either. Running 3a first with an oracle fitness (which trivially handles any valid trellis) cleanly isolates risk #1. Once 3a converges, 3b layers on exactly one new variable: the decoder.

**Caveat for 3a's scientific scope.** Under oracle decoding, interference is perfectly subtracted, so the effective channel is AWGN. Phase 3a therefore reduces to *"find the best 64-state rate-1/2 code for AWGN under ML decoding"* — a classically studied problem where NASA K=7 (Odenwalder) is already near-optimal. Do not expect 3a to produce a novel code. Its value is **(i) validating the EA converges, and (ii) establishing the oracle-decoding ceiling against which 3b's co-designed trellis is measured.** If 3b's best trellis ≠ 3a's best trellis, *that* is the evidence for co-design mattering.

**Simulated annealing is deferred.** SA is a fallback for when the EA plateaus at a clearly-suboptimal fitness. Not implemented in 3a or 3b. Will be added later (~2 days of work) only if EA diagnostics indicate deception or local-optima trapping.

---

#### Shared Foundation (build before 3a, reuse in 3b) — not yet started

**These modules must exist, be unit-tested, and be frozen before either 3a or 3b runs.**

- [ ] `trellis_genome.py`:
  - [ ] Candidate representation: `next_state: (64, 2) int`, `output_pair: (64, 2) int ∈ {0,1,2,3}` where 0=(−1,−1), 1=(−1,+1), 2=(+1,−1), 3=(+1,+1)
  - [ ] `serialize(trellis) -> bytes` / `deserialize(b) -> trellis` for reproducible hashing
  - [ ] `genome_hash(trellis) -> str` — SHA-256 of serialized bytes, used as the logging key
  - [ ] `perturb(trellis, n_edges, rng) -> trellis` — single-edge rewirings and output-pair flips
  - [ ] `random_valid_trellis(rng, max_tries) -> trellis` — seed uniformly over validity-filtered space
  - [ ] `nasa_k7_trellis() -> trellis` — reference (133,171)₈
- [ ] `constraints.py`:
  - [ ] `is_non_catastrophic(trellis) -> bool` — state-pair difference-graph test (Lin & Costello Ch. 11)
  - [ ] `is_fully_connected(trellis) -> bool` — BFS from state 0 reaches all 64 states AND from every state, state 0 is reachable (required for termination)
  - [ ] `compute_dfree(trellis) -> int` — modified state diagram / Dijkstra on error trellis
  - [ ] **Required unit test:** `compute_dfree(nasa_k7_trellis())` must return exactly 10. Do not proceed to the EA loop until this passes.
- [ ] `search.py`:
  - [ ] Population management (pop=50, elite=2)
  - [ ] Tournament selection (size 3)
  - [ ] Mutation (1–3 edges per genome, reject-and-retry if constraint-violating)
  - [ ] State-wise crossover with repair step (swap all outgoing edges from a random subset of states)
  - [ ] Accepts `fitness_fn: Callable[[Trellis, int], float]` — 3a and 3b plug in different fitness functions
  - [ ] Elitism-based termination: plateau detection (no best-fitness improvement in 50 generations) OR fixed gen budget (200)
  - [ ] **Deterministic logging:** every evaluated candidate's genome hash, fitness, generation, and RNG seed saved to `.npz`

**Tests for the shared foundation (must all pass before Phase 3a runs):**
- [ ] `test_constraints.py`:
  - [ ] `compute_dfree(NASA_K7) == 10`
  - [ ] `is_non_catastrophic(NASA_K7) == True`
  - [ ] `is_fully_connected(NASA_K7) == True`
  - [ ] `compute_dfree` of a known catastrophic code returns `inf` or raises
- [ ] `test_genome.py`:
  - [ ] `deserialize(serialize(T)) == T` for random valid T
  - [ ] `perturb(T, n_edges=1)` produces a valid candidate ≥ 50% of the time (if not, perturbation is too aggressive)
  - [ ] `genome_hash` is stable across runs
- [ ] `test_search_convergence.py`:
  - [ ] With a trivial synthetic fitness (e.g., Hamming distance of the genome to NASA K=7), EA converges in ≤ 50 generations. Tests elitism, selection, and mutation mechanics before any expensive fitness is plugged in.

---

#### Phase 3a — EA Search with Oracle Fitness (Target: 2–3 weeks) — not yet started

**Purpose:**
1. Validate the EA machinery end-to-end under a fast, deterministic fitness signal.
2. Establish the *oracle-decoding ceiling* — the best trellis a search can find when the decoder has perfect interference knowledge. This is the reference point for 3b.

**Tasks:**
- [ ] `fitness.py :: fitness_oracle(trellis, seed) -> float`:
  - 1,000 MC blocks with per-block randomized channel params (SNR, INR, P, φ) drawn from the training distribution
  - Decoder: reuse `branch_metric_oracle` from `decoders.py` (B2), parameterized on the candidate trellis
  - Return: BLER (lower is better)
  - Expected wall-clock: ~1 s per candidate on CPU
- [ ] Initialization: perturb NASA K=7 until the population of 50 all pass constraint checks
- [ ] Run 5 independent RNG seeds, 200 generations each
- [ ] Per-seed artifacts: `results/phase3a/best_trellis_seed{0..4}.npz`, `fitness_curves_seed{0..4}.npz`
- [ ] Aggregate figure: fitness-improvement curves (best-of-generation, mean ± std across seeds)

**Exit criteria for Phase 3a:**
1. Fitness curve (best-of-generation, mean across 5 seeds) is monotonically non-increasing (elitism guarantees this) AND drops *meaningfully* from gen 0 to gen 200. A flat curve means the EA is broken or the neighborhood around K=7 is already saturated — diagnose before proceeding.
2. Best trellis from each seed satisfies non-catastrophic + connected + d_free ≥ 8.
3. Best trellis achieves BLER *within ~0.2 dB of NASA K=7* under oracle decoding. A much larger gain is a **red flag** — either a measurement bug or a genuinely novel finding that must be verified before being trusted.
4. Seed-to-seed variance tight: coefficient of variation < 10% on final fitness.

**Deliverable:** Fitness improvement curves across seeds; best trellis(es) from 3a; written confirmation that EA machinery works. This is the green light to start 3b.

---

#### Phase 3b — EA Search with N2 Fitness (Target: 1–2 weeks + optional retraining) — not yet started

**Purpose:** Discover trellises co-designed with N2. This is the core research contribution (the S1 system).

**Prerequisite sanity check (DO BEFORE writing `fitness_n2`):**

- [ ] `test_n2_generalization.py`:
  - Load `results/phase2b/checkpoints/best_model_seed32.pt`
  - Generate 10 random valid 64-state rate-1/2 trellises
  - For each, run 1,000 MC blocks at SNR=5 dB, INR=5 dB using `viterbi_neural_bm` with the candidate's output-pair-index table
  - Record the BLER distribution

**Decision tree based on median BLER:**
- **Median BLER < 0.5** → N2 generalizes across trellises. Proceed with frozen N2 as 3b fitness decoder.
- **Median BLER ∈ [0.5, 0.9]** → N2 partially generalizes. Two options: (a) meta-train a new N2 on a *distribution* of random valid trellises (~1 week extra work) to improve generalization, or (b) accept a noisier fitness signal and run EA with larger populations.
- **Median BLER > 0.9** → N2 does not generalize. 3b as designed is not viable; revisit plan before writing `search.py` changes.

Do not skip this. Without it, a flat 3b fitness curve is indistinguishable from a bug.

**Tasks:**
- [ ] `fitness.py :: fitness_n2(trellis, seed) -> float`:
  - 1,000 MC blocks, same channel distribution as 3a
  - Decoder: frozen seed-32 N2 + `viterbi_neural_bm`, with candidate trellis's output-pair-index table swapped in
  - Return: BLER
  - Expected wall-clock: ~1–2 s per candidate (one NN forward pass + Viterbi ACS per block, batched)
- [ ] Run 5 independent RNG seeds, 200 generations each, reusing `search.py` from the shared foundation
- [ ] Per-seed artifacts: `results/phase3b/best_trellis_seed{0..4}.npz`, `fitness_curves_seed{0..4}.npz`
- [ ] **Tier 2 (final only, not during search):** For the top 5 candidates from the final generation (aggregated across seeds), run 10,000 MC trials across SNR ∈ {0,2,4,6,8,10} dB × INR ∈ {-5,0,5,10} dB. Optional: short N2 fine-tune per candidate (~100 epochs on that trellis specifically) to see how much additional gain retraining adds.
- [ ] Save Tier 2 output to `results/phase3b/tier2_sweep.npz`

**Exit criteria for Phase 3b:**
1. Search converges (same monotone-non-increasing curve shape as 3a).
2. **Primary success bar:** at BLER=10⁻³, `3b_best + N2` beats `random_trellis + N2` (B3) by **≥ 1 dB**.
3. **Co-design evidence:** at BLER=10⁻³, `3b_best + N2` ≥ `3a_best + N2` by ≥ 0.5 dB. If not, N2 adds no co-design gain beyond "pick a good classical code" — still a publishable negative result but shifts the narrative.
4. **Headline figure for the paper:** BLER vs SNR at INR=5 dB, four curves —
   - (i) K=7 + B1 (mismatched Viterbi)
   - (ii) K=7 + N2 (Phase 2b result)
   - (iii) random trellis + N2 (B3)
   - (iv) 3b-best + N2 (S1)
   Story: (i)→(ii) is the Phase 2b decoder-side gain; (ii)→(iv) is the Phase 3 code-side gain; (iii) << (iv) proves the search is non-trivial.

**Deliverable:** Candidate trellis(es); 4-curve headline figure; comparison of 3a-best vs 3b-best to demonstrate co-design effect.

---

### Phase 4 — Analysis, Ablations, and Paper (Target: ~2–4 weeks)
**Goal:** Understand *why* joint co-design helps, and write it up clearly.

**Ablation matrix (all at equal 2.1M compute budget):**

| System | Trellis | Decoder | Purpose |
|--------|---------|---------|---------|
| B1 | K=7 classical | Mismatched Viterbi | Primary baseline |
| B2 | K=7 classical | Oracle Viterbi | Upper bound |
| B5 | K=7 classical | IC + Viterbi | DSP cancellation baseline |
| ~~N1~~ | ~~K=7 classical~~ | ~~GRU end-to-end~~ | ~~Phase 2a negative result — excluded from final ablation~~ |
| N2 | K=7 classical | Neural BM + Viterbi | Phase 2b structured hybrid |
| S1 | **Searched (3b)** | **Neural BM (N2)** | **Core result** |
| S1-oracle | Searched (3a) | Oracle Viterbi | Phase 3a intermediate; oracle ceiling |
| B3 | Random valid | Neural BM (N2) | Validates search is non-trivial (≥ 1 dB gap to S1) |

**Stress tests:**
- [ ] Shift INR by +5 dB beyond training — graceful degradation?
- [ ] Shift P outside [8,32] range — generalization?
- [ ] Test SNR range [0,10] vs [5,15] — distribution shift?

**Interpretability:**
- [ ] Which error events dominate? Analyze error patterns
- [ ] Does searched trellis show implicit interleaving structure?
- [ ] Does neural decoder exploit periodicity of interference?
- [ ] Compare state-transition matrices: 3a-best vs 3b-best vs NASA K=7. Is there structural divergence between 3a and 3b, or do they find the same code?

**Paper writeup checklist:**
- [ ] System model section (channel, trellis, decoder)
- [ ] Baselines section (B1–B5 with compute table)
- [ ] Neural approaches section (N1 negative result, N2 architecture diagram)
- [ ] Search algorithm section (shared foundation + 3a/3b decomposition)
- [ ] Results: BLER curves, compute-matched comparisons, ablation table
- [ ] Discussion: when does each approach win, and why

---

## Success Criteria

| Phase | Criterion |
|-------|-----------|
| Phase 1 | Validation tests pass; baseline BLER curves match theory |
| Phase 2a | ~~N1 beats B1 by ≥0.5 dB~~ — **negative result**: BiGRU end-to-end cannot decode K=7 code; pivoted to N2 (neural branch metric) |
| Phase 2b | N2 beats B1 by ≥0.5 dB at BLER=10⁻³; N2 shows robustness under distribution shift |
| Phase 3a | Oracle-fitness EA converges; fitness curve monotone non-increasing across 5 seeds; best trellis d_free ≥ 8; within ~0.2 dB of NASA K=7 under oracle decoding |
| Phase 3b | N2-generalization sanity check passes (median BLER < 0.5 on 10 random trellises); N2-fitness EA converges; 3b-best + N2 beats B3 by ≥ 1 dB; 3b-best + N2 ≥ 3a-best + N2 by ≥ 0.5 dB (co-design effect) |
| Phase 4 | S1 beats B1 at equal 2.1M op budget; all ablations complete; compute costs reported for every method |

---

## Failure Modes to Avoid
- Neural decoder worse than mismatched Viterbi → core thesis fails
- Searched trellis ≈ random trellis → search not working
- System only works at one SNR/INR point → overfitting
- Any comparison made without matching compute budget → unfair, rejected by reviewers
- **Phase 3 specific:** d_free implementation buggy → EA explores invalid subspace silently; always unit-test `d_free(NASA K=7) == 10` before running any search
- **Phase 3b specific:** skipping N2-generalization prerequisite → flat fitness curve is indistinguishable from EA bug
- **Phase 3a/3b comparison:** B3 (random trellis) and S1 (searched) must use the **same decoder (N2)** — a mismatch makes the ≥1 dB gap meaningless

---

## Coding Conventions
- All functions typed with Python type hints
- No global state
- **Never hardcode N=524 or K=256 as literals.** Derive coded block length as `N = 2 * (K + memory)` where `memory = K_c - 1 = 6`. Use config dicts or function parameters.
- Every module has a `__main__` block with a basic smoke test
- Random seeds: always pass `seed` argument explicitly, never rely on global state
- Cluster-ready: no hardcoded local paths; use `pathlib.Path` and config dicts
- Save all results as `.npz` + `.json` (not just plots) for reproducibility
- **Phase 3 specific:** every evaluated candidate trellis must be logged with its `genome_hash` (SHA-256 of serialized bytes), fitness value, generation index, and RNG seed

---

## Future Extensions (Do Not Implement Now)
- Fading on top of AWGN (professor mentioned)
- Structured search over polynomial-generated trellises (professor mentioned)  
- Rate 2/3 generalization
- Quantized LLRs / reduced precision
- Simulated annealing as alternative search (deferred; ~2 days of work if EA plateaus)
- Meta-trained N2 on a distribution of random trellises (deferred unless Phase 3b prerequisite fails)

## Skills
- Simulation code: read `.claude/skills/simulation/SKILL.md` before writing channel/trellis/decoder code
- Evaluation: read `.claude/skills/evaluation/SKILL.md` before writing eval or compute-cost code
- Plotting: read `.claude/skills/plotting/SKILL.md` before writing plotting or figure-generation code