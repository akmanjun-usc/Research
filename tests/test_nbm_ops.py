"""
test_nbm_ops.py — Verify Neural Branch Metric (N2) compute budget

Total ops = BiGRU RNN ops + Linear head ops + Viterbi ACS ops.
Must be ≤ 2.1M (project-wide constraint).

Note: count_flops_birnn_analytical's linear_flops is calibrated for the
Phase 2a linear head (2h → K_INFO per block).  For Phase 2b the linear
head is 2h → 4 per time step, so we compute ops manually.
"""

import pytest
from neural_bm import NeuralBranchMetric
from compute_cost import OP_BUDGET, count_flops_viterbi
from trellis import K_INFO, CONSTRAINT_LEN

# Trellis parameters derived from project constants
T = K_INFO + CONSTRAINT_LEN - 1   # 262 trellis steps
N_CODED_ACTUAL = 2 * T             # 524 coded symbols
N_STATES = 64
INPUT_DIM = 2                      # paired samples per step


def _bigru_rnn_ops(hidden_size: int) -> int:
    """
    GRU RNN operations for bidirectional, 1-layer, input_dim=2.

    Per direction per step:  6 * h * (h + d)   (3 gates × 2 ops)
    Two directions, T steps: 2 * T * 6 * h * (h + d)
    """
    h, d, n_dir = hidden_size, INPUT_DIM, 2
    return n_dir * T * 6 * h * (h + d)


def _linear_head_ops(hidden_size: int, num_outputs: int = 4) -> int:
    """
    Linear head ops: (2*h → num_outputs) applied at each of T time steps.
    """
    return T * (2 * hidden_size) * num_outputs


@pytest.mark.parametrize("hidden_size", [8, 12, 16, 24])
def test_nn_ops_within_budget(hidden_size: int) -> None:
    """NN-only ops (BiGRU + linear head) must be well under budget."""
    nn_ops = _bigru_rnn_ops(hidden_size) + _linear_head_ops(hidden_size)
    assert nn_ops <= OP_BUDGET, (
        f"h={hidden_size}: NN ops {nn_ops:,} exceed budget {OP_BUDGET:,}"
    )


def test_total_ops_h16_within_budget() -> None:
    """
    Total system ops (BiGRU h=16 + Viterbi) must be ≤ 2.1M.
    Expected breakdown:
      BiGRU rnn:  ≈ 905K
      Linear head: ≈  34K
      Viterbi ACS: ≈  67K
      Total:       ≈ 1.0M  (< 2.1M budget)
    """
    hidden_size = 16
    rnn_ops = _bigru_rnn_ops(hidden_size)
    lin_ops = _linear_head_ops(hidden_size)
    viterbi_ops = count_flops_viterbi(N_STATES, N_CODED_ACTUAL, n_outputs=2)

    total = rnn_ops + lin_ops + viterbi_ops

    print(f"\n  BiGRU rnn ops:   {rnn_ops:>10,}")
    print(f"  Linear head ops: {lin_ops:>10,}")
    print(f"  Viterbi ops:     {viterbi_ops:>10,}")
    print(f"  Total:           {total:>10,}  ({total / OP_BUDGET * 100:.1f}% of budget)")

    assert total <= OP_BUDGET, (
        f"Total ops {total:,} exceed budget {OP_BUDGET:,}"
    )
    # Sanity: should be comfortably under budget (< 60%)
    assert total < 0.6 * OP_BUDGET, (
        f"Total ops {total:,} are unexpectedly close to budget — check formula"
    )


def test_model_parameter_count() -> None:
    """NeuralBranchMetric h=16 should have a small, predictable parameter count."""
    model = NeuralBranchMetric(hidden_size=16)
    n_params = sum(p.numel() for p in model.parameters())

    # BiGRU: 2 dirs × 3 gates × [h*(h+d) + h] = 2*3*[16*18+16] = 6*304 = 1,824 (weights + biases per dir)
    # Actual PyTorch GRU param count per dir: 3 * (h*(h+d)) + 3*h = 3*16*18 + 3*16 = 864 + 48 = 912
    # Two dirs: 1,824
    # BatchNorm: 2*h * 2 = 64
    # Linear: (2*h)*4 + 4 = 128 + 4 = 132
    # Total expected ≈ 1,824 + 64 + 132 = 2,020
    print(f"\n  NeuralBranchMetric h=16 parameters: {n_params:,}")
    assert 1_500 < n_params < 4_000, (
        f"Unexpected parameter count: {n_params}. Expected ~2,000."
    )


def test_viterbi_ops() -> None:
    """Viterbi ACS ops = n_states * 2 * T * n_outputs."""
    viterbi_ops = count_flops_viterbi(N_STATES, N_CODED_ACTUAL, n_outputs=2)
    expected = N_STATES * 2 * T * 2  # = 64 * 2 * 262 * 2 = 67,072
    assert viterbi_ops == expected, (
        f"Viterbi ops {viterbi_ops} != expected {expected}"
    )
    assert viterbi_ops <= OP_BUDGET
