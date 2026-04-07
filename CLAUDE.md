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
│   ├── test_neural_vs_b1.py    # [Phase 2a] N1 must beat B1 by ≥0.5 dB
│   └── test_overfit_gru_batch.py   # [Phase 2a] Single-batch GRU overfit sanity check
└── results/
    ├── phase1/         # Phase 1 eval: .npz data, compute_table.md
    │   └── figures/    # phase1_bler_vs_snr.pdf, phase1_bler_vs_inr.pdf
    ├── phase2a/        # Phase 2a eval: model checkpoints (empty — training failed), logs
    │   ├── checkpoints/  # Empty — no successful model saved
    │   └── logs/         # Training logs (if any runs completed)
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
- For classical methods: count multiply-adds analytically (Viterbi: O(S²·N), where N=524)
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

#### 2b — Neural Branch Metric Estimator (N2) ← **ACTIVE NEXT PHASE**

**Motivation (from Phase 2a negative result):** End-to-end decoding failed because a small BiGRU cannot implicitly learn Viterbi over 64 states. Solution: keep the Viterbi/BCJR graph structure intact and replace only the branch metric computation with a small neural network. The NN needs to learn only a scalar log-likelihood proxy per branch per time step — a much simpler mapping than end-to-end decoding.

**Concept:** Replace γ(s→s', y_t) in the Viterbi ACS operation with a learned NN output. The DP structure handles global search; the NN handles local channel estimation.

- [ ] Design NN architecture for branch metric estimation (see design notes above)
- [ ] Integrate with existing Viterbi/BCJR graph structure in `decoders.py`
- [ ] Verify op count matches 2.1M budget
- [ ] Compare N2 vs B1 (mismatched Viterbi) and B5 (IC+Viterbi)

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