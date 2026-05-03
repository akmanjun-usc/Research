# CLAUDE.md — Search-Designed Trellis Codes with Neural Decoding
**Course:** EE597 | **Author:** Abhishek Manjunath  
**Deliverable:** Simulation results + IEEE-style paper writeup  
**Stack:** Python 3.10+, PyTorch 2.0+, NumPy, SciPy, Matplotlib/Seaborn  
**Compute:** Local GPU/CPU; write cluster-ready code (no hardcoded paths, use `pathlib`)

---

## Project Thesis

> "Co-designing the trellis and a constrained learned decoder for the realistic impairment family produces a better end-to-end system than optimizing either independently."

Goal: robustness and graceful degradation under periodic sinusoidal interference — not beating polar codes on clean AWGN.

---

## System Parameters (Fixed)

| Parameter | Value |
|-----------|-------|
| Modulation | BPSK ∈ {-1, +1} |
| Info bits | K = 256 |
| Constraint length | K_c = 7 (memory m = 6, S = 2^6 = 64 states) |
| Tail bits | 6 zero bits → forces trellis back to state 0 |
| Total input bits | K + m = 262 |
| Coded block length | N = 2 × 262 = 524 |
| Code rate | R_eff = 256/524 ≈ 0.4885 |
| SNR range | 0–10 dB (eval focus: 3–5 dB) |
| INR range | -5 to 15 dB (eval focus: 5 dB) |
| Interference model | i[t] = A·sin(2πt/P + φ), P ∈ [8,32], φ ∈ [0,2π], A = √(2·INR_linear·σ²) |
| Target BLER eval point | BLER = 10⁻³ |
| Monte Carlo trials | 100,000 (final eval); 10,000 (search full); 1,000 (search proxy) |
| Op budget | 2.1M operations (all methods must match) |

**Never hardcode N=524 or K=256.** Derive as `N = 2 * (K + memory)` where `memory = K_c - 1 = 6`.

---

## Codebase Structure

```
project/
├── channel.py           # AWGN + sinusoidal interference
├── trellis.py           # FSM: encode, validate, load NASA K=7
├── decoders.py          # Viterbi (mismatched B1 + oracle B2); C speedup via viterbi_core.so
├── viterbi_core.c/.so   # C extension for Viterbi inner loop
├── build_viterbi.sh     # Build script for viterbi_core.so
├── interference_est.py  # FFT-based interference estimation + cancellation
├── baselines.py         # B1, B2, B3, B5 wrappers
├── eval.py              # Monte Carlo BLER, SNR/INR sweeps
├── compute_cost.py      # FLOPs/latency profiler
├── plot_utils.py        # IEEE-style figures, Paul Tol palette
├── plot_training_history.py  # Training/validation loss curves
├── neural_decoder.py    # [Phase 2a, negative result] BiRNNDecoder end-to-end
├── neural_bm.py         # [Phase 2b] NeuralBranchMetric + viterbi_neural_bm + training
├── constraints.py       # [Phase 3] Python reference: is_fully_connected, is_terminating,
│                        #   is_non_catastrophic, compute_dfree
├── trellis_genome.py    # [Phase 3] TrellisGenome TypedDict, serialize/deserialize,
│                        #   genome_hash, perturb, random_valid_trellis, mutate_and_validate
├── search.py            # [Phase 3] EA: population init, tournament, crossover, run_ea
├── fitness.py           # [Phase 3] fitness_oracle (3a) + fitness_n2 (3b, partial)
├── phase3_core.c/.so    # [Phase 3] C: encode, viterbi_neural_bm, constraint checks,
│                        #   mutate_and_validate
├── build_phase3_core.sh # Build script for phase3_core.so
├── phase3_native.py     # ctypes wrappers → phase3_core.so; falls back to Python
├── run_phase3a.py       # Phase 3a orchestrator (5-seed EA runs)
├── profile_phase3a.py   # Timing/profiling harness
├── tests/               # All test files (see Tests section)
└── results/
    ├── phase1/          # BLER curves, compute_table.md
    ├── phase2a/         # Empty checkpoints (negative result)
    ├── phase2b/         # Checkpoints, BLER curves, logs, compute tables
    ├── phase3a/         # EA logs + best trellis per seed (partial run)
    └── test/            # Test-generated plots
```

---

## Methods Summary

| ID | Trellis | Decoder | Status |
|----|---------|---------|--------|
| B1 | NASA K=7 | Mismatched Viterbi (AWGN metric) | ✅ Phase 1 complete |
| B2 | NASA K=7 | Oracle Viterbi (perfect interference knowledge) | ✅ Phase 1 complete |
| B5 | NASA K=7 | IC (FFT cancel) + Viterbi | ✅ Phase 1 complete |
| N1 | NASA K=7 | BiGRU end-to-end | ✅ Phase 2a — **negative result** (BLER=1.0) |
| N2 | NASA K=7 | Neural BM + Viterbi (seed32 checkpoint) | ✅ Phase 2b — BLER=2.50e-3 @ 5dB/5dB |
| B3 | Random valid | N2 | ⬜ Phase 3b prerequisite (not yet run) |
| S1-oracle | Searched (3a) | Oracle Viterbi | 🔄 Phase 3a — in progress |
| S1 | Searched (3b) | N2 | ⬜ Phase 3b — not started |

---

## Current State (2026-05-03)

### Completed
- **Phase 1** (B1, B2, B5): BLER curves, compute tables, all tests passing.
- **Phase 2a** (N1): Negative result documented. BiGRU end-to-end cannot decode K=7 — model in `neural_decoder.py`, kept for reproducibility.
- **Phase 2b** (N2): `neural_bm.py` with BiGRU (h=16/dir) + BN + Linear head. Seed32 checkpoint (`results/phase2b/checkpoints/best_model_seed32.pt`) is the primary model — NN learns unnormalized oracle distance, σ² scaling applied externally at decode time.
- **Phase 3 shared foundation**: All code written and tested — `trellis_genome.py`, `constraints.py`, `search.py`, `fitness.py`, `phase3_native.py`, `phase3_core.c`, `phase3_core.so`, `build_phase3_core.sh`, `run_phase3a.py`.

### Phase 3a — In Progress
`run_phase3a.py` runs a 5-seed EA with `fitness_oracle` (oracle Viterbi decoder, SNR=3dB, INR=5dB, 1000 MC trials).

| Seed | Generations completed | Best fitness |
|------|----------------------|--------------|
| 0 | 60 | 1.1e-2 |
| 1–4 | 1 (incomplete) | 0.0 (hit 0 immediately) |

Seeds 1–4 need to be re-run or diagnosed. Results in `results/phase3a/`.

### Phase 3b — Not Started
No `results/phase3b/` directory. `test_n2_generalization.py` not written yet.
`fitness_n2` in `fitness.py` is partially implemented (check before use).

---

## What's Next

1. **Diagnose / complete Phase 3a seeds 1–4**: Seeds show BLER=0 after 1 generation — likely the population was initialized with many zero-BLER trellises (NASA K=7 variants), causing the plateau trigger to fire immediately. Investigate `run_phase3a.py` plateau logic and rerun.

2. **Write `tests/test_n2_generalization.py`**: Load seed32 checkpoint, generate 10 random valid trellises, evaluate BLER with `viterbi_neural_bm` for each. Decision tree: median BLER < 0.5 → proceed with frozen N2 for 3b; otherwise revise plan.

3. **Phase 3b**: Implement `fitness_n2` fully in `fitness.py`, create `results/phase3b/`, run 5-seed EA. Exit criteria: 3b-best + N2 beats B3 by ≥1 dB; 3b-best + N2 ≥ 3a-best + N2 by ≥0.5 dB.

4. **Phase 4**: Ablations, stress tests, paper writeup.

---

## N2 Design (Phase 2b) — Key Facts

Architecture: `(batch, 262, 2)` → BiGRU(h=16/dir) → BN → Linear(32→4) → `(batch, 262, 4)` branch metric scores for the 4 BPSK output pairs {(−1,−1),(−1,+1),(+1,−1),(+1,+1)}.

Training target: unnormalized oracle distance `−‖y−x−i‖²` per output pair. σ² scaling applied externally as `bm / (2σ²)` at inference. This is what `seed32` was trained with.

**Why N2 is pluggable into Phase 3b**: The NN outputs 4 scores per time step based on *received signal only*. The trellis structure enters only through the output-pair-index lookup during Viterbi ACS. So the frozen N2 can be used with arbitrary trellises — validity of this assumption is checked in `test_n2_generalization.py`.

FLOPs: 913,664 (43.5% of 2.1M budget). Compute tables: `results/phase2b/compute_table_seed32.md`.

---

## Phase 3 Architecture — Key Facts

**Genome**: `TrellisGenome = {next_state: (64,2) int32, output_pair: (64,2) int32 ∈ {0,1,2,3}, n_states: 64}` where 0=(−1,−1), 1=(−1,+1), 2=(+1,−1), 3=(+1,+1).

**Constraint checks** (in order, cheapest first): `is_fully_connected` → `is_terminating` → `is_non_catastrophic` → `compute_dfree ≥ target`. The C extension (`phase3_core.so`) fuses all four into `mutate_and_validate_c` to avoid Python↔C boundary cost on rejections.

**Required anchor**: `compute_dfree(nasa_k7_genome()) == 10`. Both Python and C. Do not run EA if this fails.

**EA parameters**: pop=50, elite=2, tournament=3, mutation=1–3 edges, plateau patience=50 gens, max 200 gens.

**Logging**: every evaluated candidate → `genome_hash` (SHA-256), `fitness`, `generation`, `eval_seed` → `log_seedN.npz`.

---

## Compute Budget

Every method must stay within 2.1M ops. Report as a table in the paper.

| Method | FLOPs | Notes |
|--------|-------|-------|
| B1/B2 (Viterbi) | ~134K | S²·N analytical |
| N2 (BiGRU + Viterbi) | ~914K | 43.5% of budget |
| B5 (IC + Viterbi) | ~134K + FFT | |

Use `compute_cost.py` for profiling. For neural methods, use `thop`.

---

## Tests

| File | Phase | What it checks |
|------|-------|----------------|
| `test_encode_decode.py` | 1 | Noiseless round-trip |
| `test_awgn_theory.py` | 1 | BLER matches K=7 theory |
| `test_oracle_vs_mismatch.py` | 1 | Oracle ≥ mismatched always |
| `test_bler_vs_snr_plot.py` | 1 | 5-curve BLER plot |
| `test_neural_ops.py` | 2a | GRU op count ≤ 2.1M |
| `test_neural_overfit.py` | 2a | BiGRU learns on 1 batch |
| `test_neural_vs_b1.py` | 2a | N1 vs B1 |
| `test_overfit_gru_batch.py` | 2a | GRU overfit sanity |
| `test_nbm_ops.py` | 2b | N2 + Viterbi ops ≤ 2.1M |
| `test_nbm_overfit.py` | 2b | Branch metrics converge |
| `test_nbm_awgn.py` | 2b | N2 ≈ B1 on pure AWGN |
| `test_nbm_vs_b1.py` | 2b | N2 beats B1 by ≥0.5 dB |
| `test_constraints.py` | 3 | dfree(K=7)=10, connectivity |
| `test_genome.py` | 3 | Serialize round-trip, perturbation rate |
| `test_search_convergence.py` | 3 | EA converges on trivial fitness |
| `test_c_python_parity.py` | 3 | C == Python for all functions |
| `test_run_phase3a.py` | 3a | Progress table rendering |
| `test_profile_phase3a.py` | 3a | Profiling harness |
| `test_n2_generalization.py` | **3b prerequisite** | **MISSING — must write before 3b** |

---

## Success Criteria

| Phase | Criterion |
|-------|-----------|
| 1 | Validation tests pass; BLER curves match theory |
| 2a | Negative result documented |
| 2b | N2 beats B1 by ≥0.5 dB at BLER=10⁻³ ✅ |
| 3a | EA converges (fitness monotone non-increasing); best trellis d_free ≥ 8; within ~0.2 dB of K=7 under oracle decoding; CV < 10% across seeds |
| 3b | N2 generalizes (median BLER < 0.5 on 10 random trellises); 3b-best + N2 beats B3 by ≥1 dB; 3b-best + N2 ≥ 3a-best + N2 by ≥0.5 dB |
| 4 | All ablations done; compute costs reported for every method |

---

## Coding Conventions
- All functions typed with Python type hints
- No global state; always pass `seed` explicitly
- Save results as `.npz` (not just plots)
- Every module has a `__main__` smoke test
- Use `pathlib.Path`, no hardcoded paths

## Skills
- Simulation code: read `.claude/skills/simulation/SKILL.md`
- Evaluation: read `.claude/skills/evaluation/SKILL.md`
- Plotting: read `.claude/skills/plotting/SKILL.md`
