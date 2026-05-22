# Search-Designed Trellis Codes with Neural Decoding

**EE597 course project — Abhishek Manjunath**

Simulation study of search-designed trellis codes with neural decoding for
non-traditional channels, under a strict per-block compute budget.

## Abstract

Industrial-IoT and unlicensed sub-GHz wireless links share spectrum with
narrowband emissions from variable-frequency motor drives, switching power
converters, and PWM controllers, which appear at the receiver as periodic
sinusoidal interferers superimposed on thermal noise. Under such non-Gaussian
interference, the standard mismatched soft-decision Viterbi decoder for a
rate-1/2, constraint-length-7 (NASA K=7) convolutional code saturates at
block-error rate (BLER) = 1 for moderate interference levels. I study learned
and search-based co-design of the trellis and decoder under a strict 2.1 MFLOP/block
compute budget representative of edge radios. A neural branch-metric estimator
(**N2**, ≈1.86k parameters) trained against the oracle log-likelihood, plugged
into a standard Viterbi add-compare-select, achieves BLER = 2.5×10⁻³ at
SNR = INR = 5 dB — within ≈1.2 dB of the oracle decoder by log-linear
interpolation, using 43.5% of the compute budget. An end-to-end BiGRU baseline
(N1) overruns the budget by 3.17× and fails to learn, isolating the design
lesson that local channel estimation, not global trellis search, is the right
neural-component scope. An evolutionary algorithm (EA) over valid 64-state
rate-1/2 finite-state machines, run under both oracle and frozen-N2 fitness,
converges to the NASA K=7 polynomials, providing empirical evidence that this
classical code is near-optimal for the decoder class considered.

---

## 1. Problem setup

| Parameter | Value |
|-----------|-------|
| Modulation | BPSK ∈ {−1, +1} |
| Info bits | K = 256 |
| Constraint length | K_c = 7 (memory m = 6 → S = 64 trellis states) |
| Tail bits | 6 zero bits (force trellis back to state 0) |
| Coded block length | N = 2 × (K + m) = 524 |
| Code rate | R ≈ 0.4885 |
| Channel | AWGN + interference `i[t] = A·sin(2πt/P + φ)`, P ∈ [8,32], φ ∈ [0,2π] |
| Eval operating point | SNR 3–5 dB, INR 5 dB, target BLER 10⁻³ |
| Monte Carlo trials | 100,000 (final), 10,000 / 1,000 (search) |
| Compute budget | 2.1M operations — every method must fit |

Block length and state count are always **derived** (`N = 2*(K+m)`, `S = 2^m`),
never hardcoded.

---

## 2. Methods compared

| ID | Trellis | Decoder | Result |
|----|---------|---------|--------|
| **B1** | NASA K=7 | Mismatched Viterbi (AWGN metric) | Baseline |
| **B2** | NASA K=7 | Oracle Viterbi (perfect interference knowledge) | Upper bound |
| **B5** | NASA K=7 | FFT interference cancellation + Viterbi | Classical IC |
| **N1** | NASA K=7 | BiGRU end-to-end | **Negative result** — cannot decode K=7 (BLER = 1.0) |
| **N2** | NASA K=7 | Neural branch metric + Viterbi | **BLER 2.5e-3 @ 5/5 dB** — beats B1 by ≥0.5 dB |
| **B3** | Random valid trellis | frozen N2 | **Negative result** — 9× worse than N2 on NASA K=7 |
| **S1-oracle** | EA-searched | Oracle Viterbi | Converges to ≈ NASA K=7 |
| **S1** | EA-searched | N2 | Converges to ≈ NASA K=7 |

**N2** is the key idea: a small BiGRU (hidden 16/direction) reads the received
signal and outputs 4 branch-metric scores per time step (one per BPSK output
pair). The trellis enters Viterbi only through an output-pair lookup, so the same
frozen N2 plugs into *any* trellis — which is what makes the Phase 3 search
possible. N2 costs ~914K FLOPs (43.5% of the 2.1M budget).

---

## 3. Repository layout

```
Simulation core
  channel.py            AWGN + sinusoidal interference
  trellis.py            FSM encode / validate / NASA K=7 loader
  decoders.py           Viterbi: mismatched (B1) + oracle (B2)
  viterbi_core.c/.so    C extension for the Viterbi inner loop
  interference_est.py   FFT-based interference estimation + cancellation (B5)
  baselines.py          B1 / B2 / B3 / B5 wrappers

Neural decoders
  neural_decoder.py     N1 — BiGRU end-to-end (Phase 2a, negative result)
  neural_bm.py          N2 — neural branch metric + Viterbi + training loop

Trellis search (Phase 3)
  constraints.py        connectivity / termination / catastrophic / d_free checks
  trellis_genome.py     genome representation, mutation, validation
  search.py             evolutionary algorithm (population, tournament, crossover)
  fitness.py            fitness_oracle (3a) and fitness_n2 (3b)
  phase3_core.c/.so     fused C constraint checks + Viterbi
  phase3_native.py      ctypes wrappers (Python fallback if .so missing)
  run_phase3a.py        EA driver — oracle Viterbi fitness
  run_phase3b.py        EA driver — N2 decoder fitness

Evaluation & reporting
  eval.py               Monte Carlo BLER, SNR/INR sweeps
  compute_cost.py       FLOPs / latency profiler
  plot_utils.py         IEEE-style figures (Paul Tol colorblind palette)
  plot_training_history.py

build_viterbi.sh        builds viterbi_core.so
build_phase3_core.sh    builds phase3_core.so
tests/                  pytest suite (see Section 7)
results/                .npz data + plots, organized by phase
reports/report.tex      IEEE-style paper writeup (report.pdf)
```

---

## 4. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, scipy, torch, matplotlib, seaborn, pytest

# Build the C extensions (optional but ~30× faster; pure-Python fallback exists)
bash build_viterbi.sh
bash build_phase3_core.sh
```

Tested with Python 3.10+. The C extensions are committed as `.so` files for
macOS; rebuild them on other platforms with the scripts above.

---

## 5. Reproducing the results

All scripts write `.npz` data and plots into `results/<phase>/`. Random seeds are
always passed explicitly — runs are deterministic.

### Phase 1 — classical baselines (B1, B2, B5)
```bash
python eval.py --phase 1 --n-trials 100000 --inr-db 5 --seed 42
```

### Phase 2b — evaluate (or retrain) N2
The trained checkpoints are committed to the repo
(`results/phase2b/checkpoints/best_model_seed{32,42,99}.pt`), so evaluation works
without retraining. `seed32` is the primary model.
```bash
# Optional — retrain from scratch
python neural_bm.py --train --seed 32

# Evaluate the N2 decoder using the committed checkpoint
python eval.py --phase 2 --n-trials 100000 \
  --checkpoint results/phase2b/checkpoints/best_model_seed32.pt
```

### Phase 3a — EA search with oracle Viterbi fitness
```bash
python run_phase3a.py        # runs the EA, writes results/phase3a/best_trellis_seed0.npz
python eval.py --phase 3 --n-trials 100000 \
  --genome results/phase3a/best_trellis_seed0.npz
```

### Phase 3b — EA search with N2 decoder fitness
```bash
python run_phase3b.py        # writes results/phase3b/best_trellis_seed0.npz
python eval.py --phase 4 --n-trials 100000 \
  --genome results/phase3b/best_trellis_seed0.npz \
  --checkpoint results/phase2b/checkpoints/best_model_seed32.pt
```

### Compute-cost table
```bash
python compute_cost.py       # FLOPs / latency for every method
```

`eval.py` flags: `--phase {1,2,3,4}`, `--n-trials`, `--inr-db`, `--seed`,
`--snr-min/--snr-max/--snr-step`, `--checkpoint`, `--genome`.

---

## 6. Key results

Operating point INR = 5 dB unless noted.

| Method | BLER @ 3 dB | BLER @ 4 dB | BLER @ 5 dB |
|--------|-------------|-------------|-------------|
| N2 (NASA K=7) | — | — | 2.5e-3 |
| S1-oracle (Phase 3a, 100k trials) | 2.58e-2 | 1.4e-3 | 8e-5 |
| S1 (Phase 3b, 10k trials) | 6.35e-2 | 1.22e-2 | 2.5e-3 |

**Findings**

1. **N2 works.** The learned branch-metric decoder beats the classical mismatched
   Viterbi (B1) by ≥0.5 dB at BLER 10⁻³, within budget.
2. **N1 fails.** A BiGRU end-to-end decoder cannot decode a K=7 code — kept in the
   repo as a documented negative result.
3. **Search converges to NASA K=7.** Both EA variants (oracle and N2 fitness)
   found no trellis clearly better than the standard NASA K=7 code.
4. **A frozen N2 is trellis-specific (B3).** N2 is trained on NASA K=7's output
   patterns; reused on 498 random valid trellises it is 9× worse. True co-design
   would require *jointly* training the decoder with the trellis search — searching
   with a frozen decoder is not enough.

See `reports/report.pdf` for the full writeup, figures, and discussion.

---

## 7. Tests

```bash
pytest tests/ -q
```

The suite covers noiseless encode/decode round-trips, BLER-vs-theory agreement,
oracle ≥ mismatched ordering, neural-decoder op-count budgets, branch-metric
overfitting/convergence, trellis-constraint correctness (`d_free(NASA K=7) = 10`),
genome serialization, EA convergence on a trivial fitness, and C-vs-Python parity
for the native extensions.

---

## 8. Conventions

- Type hints on all functions; no global state — seeds passed explicitly.
- Results saved as `.npz`, not just plots.
- Every module has a `__main__` smoke test.
- `pathlib.Path` everywhere; no hardcoded paths — cluster-ready.

For deeper design notes see `CLAUDE.md`, `Phase0_Specification.txt`,
`Phase2a_Documentation.md`, and `project_summary.md`.
