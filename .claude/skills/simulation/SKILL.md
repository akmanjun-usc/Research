---
name: simulation
description: >
  Use this skill whenever writing or modifying ANY simulation code for the EE597
  trellis codes project. This includes channel.py, trellis.py, decoders.py, and
  any module that implements encoding, decoding, or channel modeling. Trigger on
  any request like "implement Viterbi", "write the channel model", "encode bits",
  "implement BCJR", "add interference", or "create the trellis FSM". Also trigger
  when the user asks to fix bugs in simulation code or extend an existing module.
  Always read this skill before writing any simulation code — it prevents critical
  numerical bugs and enforces project-wide consistency.
---

# Simulation Code Skill

## Module Ownership

Each file has a single responsibility. Never mix concerns across modules.

| File | Owns | Never Put Here |
|------|------|----------------|
| `channel.py` | AWGN generation, interference generation, combined channel | Encoding, decoding logic |
| `trellis.py` | FSM definition, encode, validate, load standard codes | Channel, decoder logic |
| `decoders.py` | Viterbi (mismatched + oracle), BCJR | Training loops, channel sim |
| `neural_decoder.py` | GRU architecture, training loop, op counting | Classical decoding |
| `neural_bm.py` | Neural branch metric estimator, Viterbi integration | End-to-end neural decode |
| `interference_est.py` | Interference parameter estimation, cancellation | Decoding |
| `baselines.py` | Wrappers for polar SCL and other external baselines | Core algorithm impl |

---

## File Template

Every simulation module must follow this structure exactly:

```python
"""
<module_name>.py — <one-line description>

Part of: EE597 Search-Designed Trellis Codes Project
"""

from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


# ─────────────────────────────────────────────
# Constants (project-wide fixed values)
# ─────────────────────────────────────────────
N_CODED = 512       # coded block length
K_INFO  = 256       # information bits per block
N_STATES = 64       # trellis states
RATE = 0.5          # code rate


# ─────────────────────────────────────────────
# Main implementation
# ─────────────────────────────────────────────

# ... your code here ...


# ─────────────────────────────────────────────
# Smoke test (run with: python channel.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running smoke test for <module>...")
    # minimal test that exercises the main path
    # must print PASS or raise AssertionError
    print("PASS")
```

---

## channel.py — Rules

### Interference formula — use exactly this, no variations:
```python
def generate_interference(
    n_symbols: int,
    amplitude: float,
    period: float,
    phase: float,
) -> np.ndarray:
    """i[t] = A * sin(2*pi*t/P + phi)"""
    t = np.arange(n_symbols)
    return amplitude * np.sin(2 * np.pi * t / period + phase)

def amplitude_from_inr(inr_db: float, noise_var: float) -> float:
    """A = sqrt(2 * INR_linear * sigma^2). Sinusoid power = A^2/2."""
    inr_linear = 10 ** (inr_db / 10)
    return np.sqrt(2 * inr_linear * noise_var)

def noise_var_from_snr(snr_db: float, es: float = 1.0) -> float:
    """sigma^2 = E_s / SNR_linear"""
    snr_linear = 10 ** (snr_db / 10)
    return es / snr_linear
```

### Channel randomization for training — always use this signature:
```python
def sample_channel_params(
    snr_range_db: tuple[float, float] = (0.0, 10.0),
    inr_range_db: tuple[float, float] = (-5.0, 10.0),
    period_range: tuple[int, int] = (8, 32),
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Returns dict with keys: snr_db, inr_db, period, phase, amplitude, noise_var"""
```

---

## trellis.py — Rules

### Trellis representation — always use this data structure:
```python
@dataclass
class Trellis:
    n_states: int            # S = 64
    next_state: np.ndarray   # shape (S, 2): next_state[s, input_bit]
    output_bits: np.ndarray  # shape (S, 2, 2): output_bits[s, input_bit, 0:2]
    name: str = "unnamed"

    def encode(self, info_bits: np.ndarray, init_state: int = 0) -> np.ndarray:
        """info_bits: (K,) binary array → coded_bits: (N,) binary array"""
        ...

    def validate(self) -> dict[str, bool]:
        """Returns: {non_catastrophic, fully_connected, terminated}"""
        ...
```

### Encoding — the only correct pattern:
```python
def encode(self, info_bits: np.ndarray, init_state: int = 0) -> np.ndarray:
    state = init_state
    coded = []
    for bit in info_bits:
        coded.extend(self.output_bits[state, bit])
        state = self.next_state[state, bit]
    # termination: force back to state 0
    while state != 0:
        # find a terminating bit
        for term_bit in [0, 1]:
            if self.next_state[state, term_bit] < state:
                coded.extend(self.output_bits[state, term_bit])
                state = self.next_state[state, term_bit]
                break
    return np.array(coded, dtype=np.int8)
```

### NASA K=7 code — load with these exact generator polynomials:
```python
# Octal: 171, 133  (standard NASA K=7 rate 1/2)
G = [0b1111001, 0b1011011]   # generators in binary
```

---

## decoders.py — Rules

### Critical: always work in log domain — NEVER in probability domain:
```python
# WRONG — underflows for long sequences
path_metric *= branch_prob

# CORRECT — numerically stable
path_metric += np.log(branch_prob + 1e-300)
```

### Branch metric computation — two modes, keep separate:

```python
def branch_metric_mismatched(
    received: np.ndarray,   # shape (2,) — two received symbols at time t
    expected: np.ndarray,   # shape (2,) — expected BPSK symbols {-1,+1}
    noise_var: float,
) -> float:
    """AWGN-only metric — ignores interference. Used for Baseline B1."""
    diff = received - expected
    return -0.5 * np.dot(diff, diff) / noise_var


def branch_metric_oracle(
    received: np.ndarray,   # shape (2,) 
    expected: np.ndarray,   # shape (2,)
    interference: np.ndarray,  # shape (2,) — known interference at time t
    noise_var: float,
) -> float:
    """Oracle metric — subtracts known interference. Used for Baseline B2."""
    diff = received - expected - interference
    return -0.5 * np.dot(diff, diff) / noise_var
```

### Viterbi — mandatory structure:
```python
def viterbi_decode(
    received: np.ndarray,        # shape (N, 2) — N/2 time steps, 2 outputs each
    trellis: Trellis,
    branch_metric_fn: callable,  # takes (received_pair, expected_pair, **kwargs)
    **metric_kwargs,             # passed to branch_metric_fn
) -> np.ndarray:
    """Returns decoded info bits as shape (K,) binary array."""
    S = trellis.n_states
    T = len(received)
    INF = -1e18

    # path_metrics[t, s] = best log-metric to reach state s at time t
    path_metrics = np.full((T + 1, S), INF)
    path_metrics[0, 0] = 0.0

    # backtrack table
    backtrack = np.zeros((T, S), dtype=np.int32)  # stores previous state
    input_bits = np.zeros((T, S), dtype=np.int8)  # stores input bit taken

    for t in range(T):
        for s in range(S):
            if path_metrics[t, s] == INF:
                continue
            for bit in [0, 1]:
                ns = trellis.next_state[s, bit]
                expected = (2 * trellis.output_bits[s, bit] - 1).astype(float)
                bm = branch_metric_fn(received[t], expected, **metric_kwargs)
                new_metric = path_metrics[t, s] + bm
                if new_metric > path_metrics[t + 1, ns]:
                    path_metrics[t + 1, ns] = new_metric
                    backtrack[t, ns] = s
                    input_bits[t, ns] = bit

    # traceback from state 0 at time T
    decoded = np.zeros(T, dtype=np.int8)
    state = 0
    for t in range(T - 1, -1, -1):
        decoded[t] = input_bits[t, state]
        state = backtrack[t, state]

    return decoded[:K_INFO]  # strip termination bits
```

---

## General Coding Rules

### Randomness — always explicit seeds:
```python
# WRONG
np.random.randn(100)

# CORRECT
rng = np.random.default_rng(seed=42)
rng.standard_normal(100)
```

### Type hints — always on public functions:
```python
# WRONG
def encode(bits, state=0):

# CORRECT
def encode(self, info_bits: np.ndarray, init_state: int = 0) -> np.ndarray:
```

### No global state — always pass config explicitly:
```python
# WRONG
GLOBAL_SNR = 5.0
def simulate(): ...  # uses GLOBAL_SNR

# CORRECT
def simulate(snr_db: float, ...) -> dict: ...
```

### BPSK convention — always {-1, +1}, never {0, 1} for transmission:
```python
def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """bits ∈ {0,1} → symbols ∈ {-1, +1}"""
    return 1 - 2 * bits.astype(float)

def bpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Hard decision: symbols → bits ∈ {0,1}"""
    return (symbols < 0).astype(np.int8)
```

### File paths — always use pathlib, never hardcoded strings:
```python
# WRONG
open("/home/user/project/results/data.npz")

# CORRECT
from pathlib import Path
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
```

### Saving results — always save raw arrays before plotting:
```python
# Always do this BEFORE any plt.show() or plt.savefig()
np.savez(RESULTS_DIR / "bler_results.npz",
         snr_db=snr_db_arr,
         bler=bler_arr,
         method=method_name)
```

---

## Validation Tests

Every implementation must pass these before being used in experiments.

### Test 1 — Noiseless round-trip (must be zero errors):
```python
def test_noiseless_roundtrip(trellis, decoder_fn):
    rng = np.random.default_rng(0)
    for _ in range(100):
        bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
        coded = trellis.encode(bits)
        symbols = bpsk_modulate(coded)
        decoded = decoder_fn(symbols.reshape(-1, 2), trellis,
                             branch_metric_mismatched, noise_var=1.0)
        assert np.array_equal(bits, decoded), "Noiseless round-trip FAILED"
    print("PASS: noiseless round-trip")
```

### Test 2 — Oracle always ≥ mismatched:
```python
def test_oracle_beats_mismatched(n_trials=1000):
    # oracle BER must be ≤ mismatched BER at every SNR point tested
    # any violation is a bug in the branch metric or decoder
    ...
```

### Test 3 — AWGN-only BLER matches theory (within 0.5 dB):
```python
# Run K=7 code with mismatched Viterbi on AWGN-only channel (INR=-inf)
# Compare against theoretical bound for K=7 d_free=10
# Must match within 0.5 dB at BLER=1e-2 to 1e-3
```
