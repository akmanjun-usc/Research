"""
compute_cost.py — FLOPs and latency profiler for all methods

Part of: EE597 Search-Designed Trellis Codes Project
"""

from __future__ import annotations
import numpy as np
import time
from typing import Optional


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
N_CODED = 512
K_INFO  = 256
N_STATES = 64
OP_BUDGET = 2_100_000


# ─────────────────────────────────────────────
# Analytical FLOPs counting
# ─────────────────────────────────────────────
def count_flops_viterbi(
    n_states: int,
    block_len: int,
    n_outputs: int = 2,
) -> int:
    """
    Viterbi complexity: S states x 2 inputs x T steps x n_outputs ops per BM.

    Each branch metric computation requires n_outputs multiply-adds
    (subtract + square for each output symbol).

    Args:
        n_states: number of trellis states
        block_len: total coded block length (N)
        n_outputs: number of output bits per trellis step

    Returns:
        total operation count
    """
    T = block_len // n_outputs  # time steps
    # Per branch: n_outputs subtractions + n_outputs multiplies + 1 add + comparison
    # Simplified: S * 2 * T * n_outputs
    return n_states * 2 * T * n_outputs


def count_flops_interference_est(block_len: int) -> int:
    """
    Approximate FLOPs for interference estimation + cancellation.
    FFT: O(N log N), least-squares: O(N), cancellation: O(N).
    """
    N = block_len
    fft_ops = int(N * np.log2(N)) * 5  # FFT complex operations
    ls_ops = N * 3 * 3  # least-squares with 3x3 normal equations
    cancel_ops = N * 3  # sin computation + multiply + subtract
    return fft_ops + ls_ops + cancel_ops


def count_flops_ic_viterbi(
    n_states: int,
    block_len: int,
    n_outputs: int = 2,
) -> int:
    """Total FLOPs for B5: interference estimation + Viterbi decoding."""
    return count_flops_interference_est(block_len) + count_flops_viterbi(
        n_states, block_len, n_outputs,
    )


# ─────────────────────────────────────────────
# Analytical RNN FLOPs counting
# ─────────────────────────────────────────────
def count_flops_birnn_analytical(
    hidden_size: int,
    input_dim: int,
    seq_len: int,
    cell_type: str = "GRU",
    bidirectional: bool = True,
) -> int:
    """
    Analytical FLOPs for a bidirectional RNN decoder + linear output.

    GRU: 3 gates, each gate = h*(h+d) multiply-adds -> 6*h*(h+d) per step
    LSTM: 4 gates -> 8*h*(h+d) per step
    RNN (vanilla): 1 transform -> 2*h*(h+d) per step

    Args:
        hidden_size: hidden units per direction
        input_dim: input feature dimension
        seq_len: sequence length (T=262 for this project)
        cell_type: "GRU", "LSTM", or "RNN"
        bidirectional: whether bidirectional

    Returns:
        total FLOPs (multiply-adds)
    """
    ops_per_unit = {"GRU": 6, "LSTM": 8, "RNN": 2}[cell_type]
    n_dir = 2 if bidirectional else 1
    h, d = hidden_size, input_dim

    rnn_flops = n_dir * seq_len * ops_per_unit * h * (h + d)

    # Linear output layer: (n_dir * h) multiply-adds per time step, K_INFO steps
    linear_flops = n_dir * h * K_INFO

    return rnn_flops + linear_flops


# ─────────────────────────────────────────────
# Neural FLOPs counting (when torch is available)
# ─────────────────────────────────────────────
def count_flops_neural(model, sample_input) -> int:
    """
    Returns total FLOPs for one forward pass of a PyTorch model.
    Uses thop if available, otherwise estimates from parameters.
    """
    try:
        from thop import profile as thop_profile
        import torch
        flops, _ = thop_profile(model, inputs=(sample_input,), verbose=False)
        return int(flops)
    except ImportError:
        # Fallback: estimate from parameter count (rough: 2 * params)
        total_params = sum(p.numel() for p in model.parameters())
        return total_params * 2


# ─────────────────────────────────────────────
# Latency measurement
# ─────────────────────────────────────────────
def measure_latency_ms(
    fn: callable,
    sample_input,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict:
    """
    Measure wall-clock latency of a function.

    Returns:
        dict with keys: mean_ms, std_ms
    """
    # Warmup
    for _ in range(n_warmup):
        fn(sample_input)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(sample_input)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {'mean_ms': np.mean(times), 'std_ms': np.std(times)}


# ─────────────────────────────────────────────
# Budget check
# ─────────────────────────────────────────────
def assert_within_budget(method_name: str, op_count: int) -> None:
    """Verify that a method's operation count is within the 2.1M budget."""
    if op_count > OP_BUDGET:
        raise ValueError(
            f"{method_name} uses {op_count:,} ops, exceeds budget of {OP_BUDGET:,}"
        )
    print(f"[OK] {method_name}: {op_count:,} ops ({op_count/OP_BUDGET*100:.1f}% of budget)")


# ─────────────────────────────────────────────
# Compute cost table
# ─────────────────────────────────────────────
def make_compute_table(methods: dict[str, dict]) -> str:
    """
    Generate markdown table of compute costs.

    Args:
        methods: {name: {'flops': int, 'latency_ms': float, 'bler_at_5db': float}}

    Returns:
        Markdown table string
    """
    lines = []
    lines.append("| Method | FLOPs | Latency (ms) | BLER @ 5dB | % Budget |")
    lines.append("|--------|-------|--------------|------------|----------|")
    for name, info in methods.items():
        flops = info.get('flops', 0)
        lat = info.get('latency_ms', 0.0)
        bler = info.get('bler_at_5db', float('nan'))
        pct = flops / OP_BUDGET * 100
        bler_str = f"{bler:.2e}" if not np.isnan(bler) else "N/A"
        lines.append(f"| {name} | {flops:,} | {lat:.2f} | {bler_str} | {pct:.1f}% |")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Smoke test (run with: python compute_cost.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running smoke test for compute_cost...")

    # Viterbi FLOPs
    viterbi_flops = count_flops_viterbi(N_STATES, N_CODED, n_outputs=2)
    print(f"  Viterbi (64-state, N=512): {viterbi_flops:,} FLOPs")
    assert_within_budget("B1_mismatched_viterbi", viterbi_flops)
    assert_within_budget("B2_oracle_viterbi", viterbi_flops)

    # IC + Viterbi FLOPs
    ic_flops = count_flops_ic_viterbi(N_STATES, N_CODED, n_outputs=2)
    print(f"  IC + Viterbi: {ic_flops:,} FLOPs")

    # Latency test
    def dummy_fn(x):
        return np.sum(x)
    lat = measure_latency_ms(dummy_fn, np.zeros(512), n_warmup=5, n_runs=50)
    print(f"  Dummy latency: {lat['mean_ms']:.4f} +/- {lat['std_ms']:.4f} ms")

    # Table test
    table = make_compute_table({
        "B1_mismatched_viterbi": {'flops': viterbi_flops, 'latency_ms': 0.0, 'bler_at_5db': 0.1},
        "B2_oracle_viterbi": {'flops': viterbi_flops, 'latency_ms': 0.0, 'bler_at_5db': 0.01},
        "B5_interference_cancel": {'flops': ic_flops, 'latency_ms': 0.0, 'bler_at_5db': 0.05},
    })
    print(f"\n{table}\n")

    print("PASS")
