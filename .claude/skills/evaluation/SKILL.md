---
name: evaluation
description: >
  Use this skill whenever writing or modifying evaluation, benchmarking, or
  compute-cost code for the EE597 trellis codes project. Trigger on any request
  like "run Monte Carlo simulation", "measure compute cost", "compare methods",
  "evaluate performance", "profile FLOPs", or "generate results tables".
  Always read this skill before writing any eval or compute-cost code — it
  enforces consistent result tracking and reporting standards across all
  experiments.
---

# Evaluation Skill

## Owned Files

| File | Purpose |
|------|---------|
| `eval.py` | Monte Carlo BLER estimation, SNR/INR sweeps |
| `compute_cost.py` | FLOPs counting, latency profiling for ALL methods |

---

## eval.py — Monte Carlo BLER

### Core evaluation function — always use this signature:
```python
def estimate_bler(
    encoder_fn: callable,
    decoder_fn: callable,
    channel_fn: callable,
    snr_db: float,
    inr_db: float,
    n_trials: int,
    seed: int = 0,
    period_range: tuple[int, int] = (8, 32),
) -> dict:
    """
    Returns:
        {
          'bler': float,           # block error rate
          'n_errors': int,         # number of block errors
          'n_trials': int,         # total trials run
          'ci_95': float,          # 95% confidence half-interval
          'snr_db': float,
          'inr_db': float,
        }
    """
    rng = np.random.default_rng(seed)
    n_errors = 0
    for _ in range(n_trials):
        info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
        rx = channel_fn(encoder_fn(info_bits), snr_db, inr_db, rng=rng)
        decoded = decoder_fn(rx)
        if not np.array_equal(info_bits, decoded):
            n_errors += 1

    bler = n_errors / n_trials
    # Wilson confidence interval
    ci = 1.96 * np.sqrt(bler * (1 - bler) / n_trials)
    return dict(bler=bler, n_errors=n_errors, n_trials=n_trials,
                ci_95=ci, snr_db=snr_db, inr_db=inr_db)
```

### SNR sweep — always save results immediately after each point:
```python
def sweep_snr(
    methods: dict[str, callable],   # name → decode_fn
    snr_range: np.ndarray,
    inr_db: float,
    n_trials: int,
    results_dir: Path,
    tag: str = "exp",
) -> dict[str, list[dict]]:
    """
    Saves results after EVERY SNR point to prevent data loss on crash.
    Returns nested dict: method_name → list of estimate_bler results
    """
    results = {name: [] for name in methods}

    for snr_db in snr_range:
        for name, decoder_fn in methods.items():
            r = estimate_bler(..., snr_db=snr_db, inr_db=inr_db, n_trials=n_trials)
            results[name].append(r)

            # Save immediately after each data point
            save_path = results_dir / f"{tag}_{name}_inr{inr_db:.0f}dB.npz"
            _save_results(results[name], save_path)
            print(f"  [{name}] SNR={snr_db:.1f}dB → BLER={r['bler']:.2e} "
                  f"(±{r['ci_95']:.2e})")

    return results
```

### Minimum trial counts by phase:
| Context | n_trials |
|---------|----------|
| Search proxy (fitness) | 1,000 |
| Search full eval | 10,000 |
| Phase 1/2 baseline curves | 100,000 |
| Final paper figures | 100,000 |

### Never report a BLER point estimated from fewer than 50 errors:
```python
def is_reliable(result: dict, min_errors: int = 50) -> bool:
    """BLER estimates with < 50 errors are unreliable — extend n_trials instead."""
    return result['n_errors'] >= min_errors
```

---

## compute_cost.py — FLOPs and Latency

### This is MANDATORY for every method. No comparison is valid without it.

```python
from thop import profile as thop_profile
import time

def count_flops_neural(model: torch.nn.Module, sample_input: torch.Tensor) -> int:
    """Returns total FLOPs (multiply-adds × 2) for one forward pass."""
    flops, _ = thop_profile(model, inputs=(sample_input,), verbose=False)
    return int(flops)

def count_flops_viterbi(n_states: int, block_len: int, n_outputs: int = 2) -> int:
    """
    Viterbi complexity: O(S^2 * T) branch metric computations.
    Each branch metric = n_outputs multiply-adds.
    """
    T = block_len // n_outputs  # time steps
    return n_states * 2 * T * n_outputs  # S states × 2 inputs × T steps × ops per BM

def measure_latency_ms(
    fn: callable,
    sample_input,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict:
    """Returns mean and std latency in ms per block."""
    # warmup
    for _ in range(n_warmup):
        fn(sample_input)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(sample_input)
        times.append((time.perf_counter() - t0) * 1000)

    return {'mean_ms': np.mean(times), 'std_ms': np.std(times)}

# OP BUDGET CHECK — call before any experiment
OP_BUDGET = 2_100_000

def assert_within_budget(method_name: str, op_count: int) -> None:
    if op_count > OP_BUDGET:
        raise ValueError(
            f"{method_name} uses {op_count:,} ops, exceeds budget of {OP_BUDGET:,}"
        )
    print(f"[OK] {method_name}: {op_count:,} ops ({op_count/OP_BUDGET*100:.1f}% of budget)")
```

### Compute cost table — always produce this for paper sections:
```python
def make_compute_table(methods: dict[str, dict]) -> str:
    """
    methods: {name: {'flops': int, 'latency_ms': float, 'bler_at_5db': float}}
    Returns a markdown table string for direct inclusion in paper draft.
    """
```

---

## Results Directory Convention

```
results/
├── phase1/
│   ├── bler_B1_inr5dB.npz
│   ├── bler_B2_inr5dB.npz
│   ├── bler_B5_inr5dB.npz
│   └── figures/
│       ├── phase1_bler_vs_snr.pdf
│       └── phase1_bler_vs_snr.png
├── phase2/
│   ├── bler_N1_inr5dB.npz
│   ├── training_curves_N1.npz
│   └── figures/
├── phase3/
│   ├── search_fitness_history.npz
│   └── candidate_trellises/
└── phase4/
    ├── ablation_all_methods.npz
    └── figures/
        └── main_result.pdf    ← the key paper figure
```

### npz schema — every results file must contain:
```python
np.savez(path,
    snr_db=snr_arr,          # shape (n_snr,)
    bler=bler_arr,           # shape (n_snr,)
    ci_95=ci_arr,            # shape (n_snr,)
    n_errors=errors_arr,     # shape (n_snr,)
    n_trials=trials_arr,     # shape (n_snr,)
    inr_db=float(inr_db),
    method=str(method_name),
    seed=int(seed),
    timestamp=str(datetime.now().isoformat()),
)
```

---

## Reporting Rules (Paper Tables)

### Main results table — always include these columns:
| Method | FLOPs | Latency (ms) | BLER @ SNR=5dB, INR=5dB | Δ vs B1 (dB) |
|--------|-------|--------------|--------------------------|--------------|

### Ablation table — Phase 4:
| Trellis | Decoder | FLOPs | BLER | Notes |
|---------|---------|-------|------|-------|
| K=7 classical | Mismatch Viterbi | ... | ... | B1 |
| K=7 classical | Oracle Viterbi | ... | ... | B2 upper bound |
| K=7 classical | IC + Viterbi | ... | ... | B5 |
| K=7 classical | GRU e2e | ... | ... | N1 |
| K=7 classical | Neural BM | ... | ... | N2 |
| **Searched** | **GRU e2e** | ... | ... | **S1 — main result** |
| Random | GRU e2e | ... | ... | B3 |

### dB gain calculation:
```python
def db_gain(bler_target: float, snr_method: np.ndarray, bler_method: np.ndarray,
            snr_baseline: np.ndarray, bler_baseline: np.ndarray) -> float:
    """SNR difference at a fixed BLER target. Positive = improvement."""
    snr_m = np.interp(bler_target, bler_method[::-1], snr_method[::-1])
    snr_b = np.interp(bler_target, bler_baseline[::-1], snr_baseline[::-1])
    return snr_b - snr_m  # positive means method needs less SNR
```
