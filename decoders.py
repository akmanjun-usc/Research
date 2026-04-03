"""
decoders.py — Viterbi decoder with mismatched and oracle branch metrics

Part of: EE597 Search-Designed Trellis Codes Project
"""

from __future__ import annotations
import ctypes
import warnings
from pathlib import Path
import numpy as np
from typing import Optional

from trellis import Trellis, load_nasa_k7, K_INFO, N_CODED
from channel import bpsk_modulate


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
INF = -1e18


# ─────────────────────────────────────────────
# C library loading (optional — graceful fallback)
# ─────────────────────────────────────────────
_C_LIB = None
_C_LOAD_ATTEMPTED = False


def _load_c_lib():
    """Load the compiled viterbi_core shared library. Returns None on failure."""
    global _C_LIB, _C_LOAD_ATTEMPTED
    if _C_LOAD_ATTEMPTED:
        return _C_LIB
    _C_LOAD_ATTEMPTED = True
    so_path = Path(__file__).parent / "viterbi_core.so"
    if not so_path.exists():
        return None
    try:
        lib = ctypes.CDLL(str(so_path))
        lib.viterbi_decode_c.restype = None
        lib.viterbi_decode_c.argtypes = [
            ctypes.POINTER(ctypes.c_double),   # received
            ctypes.POINTER(ctypes.c_double),   # interference (or NULL)
            ctypes.POINTER(ctypes.c_int),      # rev_src
            ctypes.POINTER(ctypes.c_int8),     # rev_bit
            ctypes.POINTER(ctypes.c_double),   # rev_exp
            ctypes.POINTER(ctypes.c_int),      # n_incoming
            ctypes.POINTER(ctypes.c_int8),     # decoded (output)
            ctypes.c_int,                      # S
            ctypes.c_int,                      # T
            ctypes.c_int,                      # max_inc
            ctypes.c_int,                      # K_INFO
            ctypes.c_double,                   # noise_var
        ]
        _C_LIB = lib
    except OSError:
        _C_LIB = None
    return _C_LIB


def _viterbi_c(
    received: np.ndarray,
    trellis: Trellis,
    noise_var: float,
    interference: Optional[np.ndarray],
    rev: dict,
) -> np.ndarray:
    """Call the C Viterbi implementation via ctypes."""
    lib = _load_c_lib()
    assert lib is not None

    S = trellis.n_states
    T = len(received)
    max_inc = rev['max_incoming']

    # Ensure contiguous C-order arrays
    rx = np.ascontiguousarray(received.ravel(), dtype=np.float64)
    rev_src = np.ascontiguousarray(rev['rev_src'], dtype=np.int32)
    rev_bit = np.ascontiguousarray(rev['rev_bit'], dtype=np.int8)
    rev_exp = np.ascontiguousarray(rev['rev_exp'].reshape(-1), dtype=np.float64)
    n_inc = np.ascontiguousarray(rev['n_incoming'], dtype=np.int32)

    if interference is not None:
        interf = np.ascontiguousarray(
            interference.reshape(-1, 2)[:T].ravel(), dtype=np.float64
        )
        interf_ptr = interf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        interf_ptr = ctypes.POINTER(ctypes.c_double)()  # NULL

    decoded = np.zeros(K_INFO, dtype=np.int8)

    lib.viterbi_decode_c(
        rx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        interf_ptr,
        rev_src.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        rev_bit.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        rev_exp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_inc.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        ctypes.c_int(S),
        ctypes.c_int(T),
        ctypes.c_int(max_inc),
        ctypes.c_int(K_INFO),
        ctypes.c_double(noise_var),
    )

    return decoded


# ─────────────────────────────────────────────
# Precomputation for fast Viterbi
# ─────────────────────────────────────────────
def _build_reverse_trellis(trellis: Trellis) -> dict:
    """
    For each destination state, find which (source, input_bit) pairs lead to it.
    For shift-register codes, each state has exactly 2 incoming branches.

    Returns dict with:
      - rev_src: (S, max_incoming) — source states for each destination
      - rev_bit: (S, max_incoming) — input bits for each destination
      - rev_exp: (S, max_incoming, 2) — expected BPSK symbols for each branch
      - n_incoming: (S,) — number of incoming branches per state
    """
    S = trellis.n_states
    expected_symbols = (1 - 2 * trellis.output_bits).astype(np.float64)

    # Count incoming branches per state
    incoming = [[] for _ in range(S)]
    for s in range(S):
        for bit in range(2):
            ns = trellis.next_state[s, bit]
            incoming[ns].append((s, bit))

    max_inc = max(len(inc) for inc in incoming)

    rev_src = np.zeros((S, max_inc), dtype=np.int32)
    rev_bit = np.zeros((S, max_inc), dtype=np.int8)
    rev_exp = np.zeros((S, max_inc, 2), dtype=np.float64)
    n_incoming = np.zeros(S, dtype=np.int32)

    for ns in range(S):
        n_incoming[ns] = len(incoming[ns])
        for j, (src, bit) in enumerate(incoming[ns]):
            rev_src[ns, j] = src
            rev_bit[ns, j] = bit
            rev_exp[ns, j, :] = expected_symbols[src, bit, :]

    return dict(
        rev_src=rev_src, rev_bit=rev_bit, rev_exp=rev_exp,
        n_incoming=n_incoming, max_incoming=max_inc,
    )


def viterbi_decode(
    received: np.ndarray,
    trellis: Trellis,
    branch_metric_fn: callable,
    **metric_kwargs,
) -> np.ndarray:
    """
    Viterbi decoder — fast vectorized implementation.

    Args:
        received: shape (T, 2) — T time steps, 2 output symbols each
        trellis: Trellis object
        branch_metric_fn: branch_metric_mismatched or branch_metric_oracle
        **metric_kwargs: noise_var (required), interference (optional)

    Returns:
        decoded info bits as shape (K,) binary array
    """
    noise_var = metric_kwargs.get('noise_var', 1.0)
    interference = metric_kwargs.get('interference', None)

    # Cache reverse trellis on the trellis object for reuse
    if not hasattr(trellis, '_rev_cache'):
        trellis._rev_cache = _build_reverse_trellis(trellis)

    # Try C implementation first, fall back to Python
    if _load_c_lib() is not None:
        return _viterbi_c(received, trellis, noise_var, interference, trellis._rev_cache)

    if not getattr(viterbi_decode, '_warned_no_c', False):
        warnings.warn(
            "viterbi_core.so not found — using Python fallback. "
            "Run 'bash build_viterbi.sh' to build the C extension for ~10-50x speedup.",
            stacklevel=2,
        )
        viterbi_decode._warned_no_c = True

    return _viterbi_vectorized(received, trellis, noise_var, interference, trellis._rev_cache)


def _viterbi_vectorized(
    received: np.ndarray,
    trellis: Trellis,
    noise_var: float,
    interference: Optional[np.ndarray] = None,
    rev: Optional[dict] = None,
) -> np.ndarray:
    """
    Fully vectorized Viterbi. Uses reverse trellis to process all destination
    states at once: for each dest state, compare its incoming branches in parallel.
    """
    S = trellis.n_states
    T = len(received)
    if rev is None:
        rev = _build_reverse_trellis(trellis)

    rev_src = rev['rev_src']        # (S, max_inc)
    rev_bit = rev['rev_bit']        # (S, max_inc)
    rev_exp = rev['rev_exp']        # (S, max_inc, 2)
    max_inc = rev['max_incoming']

    inv_2nv = -0.5 / noise_var

    pm_cur = np.full(S, INF)
    pm_cur[0] = 0.0

    backtrack = np.zeros((T, S), dtype=np.int32)
    input_bits_table = np.zeros((T, S), dtype=np.int8)

    interf_2d = interference.reshape(-1, 2) if interference is not None else None

    for t in range(T):
        rx = received[t]
        if interf_2d is not None:
            rx_eff = rx - interf_2d[t]
        else:
            rx_eff = rx

        # Compute branch metrics for all (dest, incoming) pairs at once
        # diff shape: (S, max_inc, 2)
        diff = rx_eff[np.newaxis, np.newaxis, :] - rev_exp
        bm = inv_2nv * np.sum(diff * diff, axis=2)  # (S, max_inc)

        # Path metrics of source states: (S, max_inc)
        src_pm = pm_cur[rev_src]  # (S, max_inc)

        # Total metrics: (S, max_inc)
        total = src_pm + bm

        # For each destination state, pick the best incoming branch
        best_idx = np.argmax(total, axis=1)  # (S,)
        pm_next = total[np.arange(S), best_idx]

        backtrack[t, :] = rev_src[np.arange(S), best_idx]
        input_bits_table[t, :] = rev_bit[np.arange(S), best_idx]

        pm_cur = pm_next

    # Traceback from state 0
    decoded = np.zeros(T, dtype=np.int8)
    state = 0
    for t in range(T - 1, -1, -1):
        decoded[t] = input_bits_table[t, state]
        state = backtrack[t, state]

    return decoded[:K_INFO]


# ─────────────────────────────────────────────
# Branch metric functions (kept for API compatibility)
# ─────────────────────────────────────────────
def branch_metric_mismatched(
    received: np.ndarray,
    expected: np.ndarray,
    noise_var: float,
    **kwargs,
) -> float:
    """AWGN-only metric — ignores interference. Used for Baseline B1."""
    diff = received - expected
    return -0.5 * np.dot(diff, diff) / noise_var


def branch_metric_oracle(
    received: np.ndarray,
    expected: np.ndarray,
    noise_var: float,
    interference: Optional[np.ndarray] = None,
    time_step: int = 0,
    **kwargs,
) -> float:
    """Oracle metric — subtracts known interference. Used for Baseline B2."""
    if interference is None:
        return branch_metric_mismatched(received, expected, noise_var)
    t_start = time_step * 2
    interf = interference[t_start:t_start + 2]
    diff = received - expected - interf
    return -0.5 * np.dot(diff, diff) / noise_var


# ─────────────────────────────────────────────
# Smoke test (run with: python decoders.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running smoke test for decoders...")
    rng = np.random.default_rng(42)

    trellis = load_nasa_k7()

    # Noiseless round-trip test
    n_pass = 0
    for i in range(20):
        info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
        coded = trellis.encode(info_bits)
        symbols = bpsk_modulate(coded)
        received = symbols.reshape(-1, 2)
        decoded = viterbi_decode(received, trellis, branch_metric_mismatched, noise_var=1.0)
        if np.array_equal(info_bits, decoded):
            n_pass += 1
        else:
            first_diff = np.where(info_bits != decoded)[0][0]
            print(f"  Trial {i}: FAIL at bit {first_diff}")

    print(f"  Noiseless round-trip: {n_pass}/20 passed")
    assert n_pass == 20, f"Noiseless round-trip failed: only {n_pass}/20 passed"

    # Test with noise
    from channel import awgn_channel, noise_var_from_snr, generate_interference, amplitude_from_inr
    info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
    coded = trellis.encode(info_bits)
    symbols = bpsk_modulate(coded)
    rx = awgn_channel(symbols, snr_db=10.0, inr_db=0.0, period=16.0, phase=0.0, rng=rng)
    received = rx.reshape(-1, 2)

    nv = noise_var_from_snr(10.0)
    decoded_mm = viterbi_decode(received, trellis, branch_metric_mismatched, noise_var=nv)
    print(f"  Mismatched: {np.sum(info_bits != decoded_mm)} bit errors")

    interf = generate_interference(len(symbols), amplitude_from_inr(0.0, nv), 16.0, 0.0)
    decoded_or = viterbi_decode(received, trellis, branch_metric_oracle, noise_var=nv, interference=interf)
    print(f"  Oracle: {np.sum(info_bits != decoded_or)} bit errors")

    # Timing test: C vs Python
    import time
    n_runs = 200

    # Time whichever backend is active (should be C if built)
    backend = "C" if _load_c_lib() is not None else "Python"
    t0 = time.perf_counter()
    for _ in range(n_runs):
        viterbi_decode(received, trellis, branch_metric_mismatched, noise_var=nv)
    elapsed_active = (time.perf_counter() - t0) / n_runs * 1000
    print(f"  Viterbi ({backend}): {elapsed_active:.3f} ms/block")

    # Also time Python fallback for comparison
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _viterbi_vectorized(received, trellis, nv, None, trellis._rev_cache)
    elapsed_py = (time.perf_counter() - t0) / n_runs * 1000
    print(f"  Viterbi (Python): {elapsed_py:.3f} ms/block")

    if backend == "C":
        print(f"  Speedup: {elapsed_py / elapsed_active:.1f}x")

    print("PASS")
