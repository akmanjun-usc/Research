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
| Block length | K=256 info bits, N=512 coded bits |
| Code rate | R = 1/2 |
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
├── channel.py          # AWGN + sinusoidal interference simulator
├── trellis.py          # Trellis FSM: encode, validate, load standard codes
├── decoders.py         # Viterbi (mismatched + oracle), optional C speedup
├── interference_est.py # FFT-based interference estimation + cancellation
├── baselines.py        # B1, B2, B5 baseline wrappers
├── eval.py             # Monte Carlo BLER, SNR/INR sweeps, Phase 1 runner
├── compute_cost.py     # FLOPs/latency profiler (applies to ALL methods)
├── plot_utils.py       # IEEE-style figures, Paul Tol palette, BLER curves
├── neural_decoder.py   # [Phase 2a] BiGRUDecoder, training loop, decode fn
├── neural_bm.py        # [Phase 2b] Neural branch metric estimator — not yet created
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
│   └── test_neural_vs_b1.py    # [Phase 2a] N1 must beat B1 by ≥0.5 dB
└── results/
    ├── phase1/         # Phase 1 eval: .npz data, compute_table.md
    │   └── figures/    # phase1_bler_vs_snr.pdf, phase1_bler_vs_inr.pdf
    ├── phase2a/        # Phase 2a eval: model checkpoints, training curves, BLER data
    │   ├── checkpoints/  # best_gru.pt, best_lstm.pt, best_rnn.pt
    │   ├── figures/      # phase2a_bler_vs_snr.pdf, phase2a_bler_vs_inr.pdf, training_loss.pdf
    │   └── logs/         # training_log.json (loss, val_bler per epoch)
    └── test/           # Test-generated plots
```

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
- For classical methods: count multiply-adds analytically (Viterbi: O(S²·N))
- Report as a table in the paper: method vs BLER vs ops vs latency

```python
# compute_cost.py must expose:
def count_ops(model_or_fn, sample_input) -> int: ...
def measure_latency(model_or_fn, sample_input, n_runs=1000) -> float: ...  # ms per block
```

---

## Neural Branch Metric Estimator — Design Notes (Phase 2b, Ideation Needed)

**Concept:** Keep the Viterbi/BCJR graph structure. Replace the branch metric γ(s→s', y_t) with a small neural network output.

**Why this is interesting:** Preserves the optimality structure of dynamic programming while learning to handle interference in the metric computation. The NN only needs to predict a scalar (log-likelihood proxy) per branch per time step.

**Architecture directions to explore (TBD with collaborator/Claude):**
- Small MLP: input = [y_t, t mod P_est, context window y_{t-k:t}] → scalar branch score
- 1D CNN over local window of received samples → branch scores for all states at time t
- Key constraint: must keep total ops ≤ 2.1M across the full decoding pass

**Open question:** How much context window does the NN need to effectively estimate interference? This is an experiment to run.

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

### Phase 2 — Neural Decoder Baselines (Target: ~2–4 weeks)
**Goal:** Validate that learned decoding improves robustness under sinusoidal interference.

**Sub-phases:**
#### 2a — GRU End-to-End Decoder (N1)

**Architecture (Bidirectional GRU)**

The decoder replaces the entire Viterbi algorithm. It takes the raw received signal and outputs bit probability estimates directly.

| Layer | Detail |
|-------|--------|
| Input | r[t] at each of 512 time steps. Input dim = 1 (optionally 2 if appending normalized time index t/512) |
| RNN | Bidirectional GRU, h=32 hidden units per direction. Forward GRU reads t=0..511, backward GRU reads t=511..0. Separate learned weights per direction. |
| Concatenation | At each position t: c[t] = [h_forward[t]; h_backward[t]], dim = 64 |
| Batch norm | Optional BatchNorm1d between GRU and output layer for training stability |
| Output | Shared linear layer W_out (1 x 64) + bias, applied at each of 512 positions, followed by sigmoid. Produces 512 probabilities. |
| Bit selection | Select 256 outputs corresponding to info bit positions (defined by trellis mapping). Discard the other 256. |

**RNN Cell Variants to Compare**

Run all three at matched compute budgets and report in paper:

| Variant | Hidden per direction | Approx ops | Purpose |
|---------|---------------------|------------|---------|
| Bidirectional GRU | 32 | ~3.2M (needs trimming, see note) | Primary candidate. GRU converges faster, fewer params than LSTM. |
| Bidirectional LSTM | 26 | ~3.2M | Compare. LSTM has separate cell state for tracking periodicity. 33% more ops per unit so smaller hidden size at same budget. |
| Bidirectional vanilla RNN | 32 | ~2.1M | Sanity check. Expected to fail on length-512 sequences due to vanishing gradients. |

**Compute Budget Note:** Bidirectional GRU with h=32 gives ~3.2M ops, exceeding the 2.1M budget. Options to resolve: (a) reduce to h=26 per direction (~2.1M ops), (b) process in temporal chunks of length 128 with overlap, (c) use unidirectional GRU h=40 (~2.0M ops) and accept loss of future context. Run op counter via `thop` to find the exact hidden size that hits 2.1M. Report final architecture choice with measured op count.

**GRU Equations (per direction, per time step)**

```
Reset gate:     r_t = sigmoid( W_r . [h_{t-1}, input_t] + b_r )
Update gate:    z_t = sigmoid( W_z . [h_{t-1}, input_t] + b_z )
Candidate:      h_tilde = tanh( W . [r_t * h_{t-1}, input_t] + b )
New state:      h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
```

Where `*` is element-wise multiplication, `[,]` is concatenation, sigmoid squashes to [0,1], tanh squashes to [-1,+1].

**Output Layer Math**

At each position t, the concatenated hidden state c[t] (64 dims) maps to one probability:
```
p[t] = sigmoid( W_out . c[t] + b_out )
```
W_out is 1x64, b_out is scalar. Same weights shared across all 512 positions (weight sharing).

Final output: select the 256 positions corresponding to info bits (positions 0, 2, 4, ..., 510 for rate-1/2 trellis, or whatever mapping trellis.py defines).

**Parameter Count**

Per GRU direction (h=32, input=1): 3 gates x (32x33 + 32) = 3264 params. Two directions = 6528. Output layer = 65. Total ~ 6593 trainable parameters.

**Loss Function**

Binary cross-entropy (BCE), averaged over K=256 info bits per block:
```
L = -(1/K) * sum_{k=0}^{K-1} [ b[k]*log(p[k]) + (1-b[k])*log(1-p[k]) ]
```
Where b[k] is the true info bit, p[k] is the predicted probability. BCE is smooth and differentiable (required for gradient-based training). Confident wrong predictions are penalized exponentially harder than uncertain ones.

**Optimizer**

Adam optimizer with the following schedule:
- Initial learning rate: 1e-3
- Reduce to 1e-4 after 50 epochs (or use ReduceLROnPlateau with patience=10)
- Gradient clipping: clip_norm=1.0 to prevent exploding gradients
- Batch size: 128 blocks per batch (tune based on GPU memory)

**Training Data Generation (on-the-fly, no pre-generated dataset)**

Each mini-batch generates fresh random data using existing channel.py and trellis.py:
1. Sample K=256 random info bits
2. Encode through trellis (trellis.py) to get N=512 coded bits
3. BPSK modulate: x[t] = 2*c[t] - 1
4. Sample channel params uniformly per block:
   - SNR ~ Uniform(0, 10) dB
   - INR ~ Uniform(-5, 10) dB (note: training INR upper bound is 10, not 15, to test generalization at 10-15 dB)
   - P ~ Uniform(8, 32) (integer or continuous, experiment with both)
   - φ ~ Uniform(0, 2π)
5. Generate noise + interference via channel.py
6. Received signal r[t] = x[t] + n[t] + i[t], shape (batch_size, 512, 1)

**Evaluation Metric**

Primary: Block Error Rate (BLER). A block is in error if ANY of the 256 decoded bits differs from the true info bits.
```
BLER = (blocks with ≥1 bit error) / (total blocks tested)
```
BLER is NOT used for training (not differentiable). It is computed during evaluation only.

Secondary (for debugging): Bit Error Rate (BER) = total wrong bits / total bits.

Evaluation uses 100,000 Monte Carlo blocks per (SNR, INR) point.

**Training Procedure**

1. Epoch = 10,000 batches of 128 blocks = 1.28M training blocks per epoch
2. Train for 100 epochs (total ~128M blocks seen)
3. Validate every epoch on held-out set: 10,000 blocks at (SNR=5dB, INR=5dB)
4. Save best model checkpoint (lowest validation BLER)
5. Early stopping: if validation BLER does not improve for 20 epochs, stop

**Implementation Tasks**

- [ ] `neural_decoder.py`: Define `BiGRUDecoder(nn.Module)` with configurable hidden_size, cell_type (GRU/LSTM/RNN), bidirectional flag, input_dim
- [ ] `neural_decoder.py`: Define `training_step(model, batch) -> loss` and `decode(model, received_signal) -> bit_estimates`
- [ ] `neural_decoder.py`: Define `train_loop(model, config, trellis, channel_fn) -> trained_model` with on-the-fly data generation
- [ ] Verify op count with `compute_cost.py` using `thop.profile()`. Adjust hidden_size until ops ≤ 2.1M
- [ ] Train GRU, LSTM, and vanilla RNN variants. Compare BLER curves.
- [ ] Evaluate: BLER vs SNR sweep at INR=5dB, BLER vs INR sweep at SNR=5dB
- [ ] Generalization test: evaluate at INR=12dB and INR=15dB (outside training range of [-5,10])
- [ ] Generalization test: evaluate at P=4 and P=48 (outside training range of [8,32])
- [ ] Compare N1 vs B1 (mismatched Viterbi) and B5 (IC+Viterbi). N1 must beat B1 by ≥0.5 dB at BLER=10⁻³.
- [ ] Save trained model weights, training curves (loss vs epoch), and all eval results to `results/phase2a/`

**Key References**

- Gruber et al., "On Deep Learning-Based Channel Decoding" (2017) — foundational RNN decoder for conv codes
- Jiang et al., "LEARN Codes: Inventing Low-Latency Codes via RNNs" (2019, IEEE TCOM) — GRU encoder/decoder co-design, robustness under channel mismatch
- Kim et al., "Communication Algorithms via Deep Learning" (2018) — Bi-GRU decoder with BatchNorm, deepcomm.github.io reference implementation
- IEEE ICC 2024, "Deep Learning Based Decoder for Concatenated Coding over Deletion Channels" — Bi-GRU as LLR estimator, single network across channel params

#### 2b — Neural Branch Metric Estimator (N2) *(Professor addition — ideation phase)*
- [ ] Design NN architecture for branch metric estimation (see design notes above)
- [ ] Integrate with existing Viterbi/BCJR graph structure in `decoders.py`
- [ ] Verify op count matches budget
- [ ] Compare N2 vs N1 vs B1 — is structured hybrid better than end-to-end?

**Deliverable:** BLER curves + compute table for N1, N2, B1, B2, B5. Evidence of robustness to distribution shift.

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
| Phase 2 | N1 beats B1 by ≥0.5 dB at BLER=10⁻³; training stable |
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