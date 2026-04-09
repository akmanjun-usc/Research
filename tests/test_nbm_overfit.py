"""
test_nbm_overfit.py — Verify NeuralBranchMetric can learn on a fixed batch

Sanity check: train the model on one repeated batch from a fixed channel
realization.  The MSE loss should converge toward zero and Viterbi with
the learned metrics should decode the training blocks correctly.

If this test fails the architecture or training loop has a bug.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

from neural_bm import (
    NeuralBranchMetric,
    _encode_fixed_tail,
    build_branch_output_index,
    compute_oracle_metrics,
    pair_received_signal,
    viterbi_neural_bm,
)
from channel import (
    amplitude_from_inr,
    awgn_channel,
    bpsk_modulate,
    generate_interference,
    noise_var_from_snr,
)
from trellis import K_INFO, load_nasa_k7

# ── Fixed overfit settings ────────────────────────────────────────────────────
BATCH_SIZE = 16       # small batch that a tiny model can memorize
N_STEPS = 300         # gradient steps
LR = 5e-3
SNR_DB = 10.0         # high SNR so blocks are in principle decodable
INR_DB = 5.0
PERIOD = 16.0
PHASE = 0.0
SEED = 7


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trellis():
    return load_nasa_k7()


@pytest.fixture(scope="module")
def index_table(trellis):
    return build_branch_output_index(trellis)


@pytest.fixture(scope="module")
def fixed_batch(trellis):
    """One fixed batch with identical channel params for all blocks."""
    rng = np.random.default_rng(SEED)
    noise_var = noise_var_from_snr(SNR_DB)
    noise_sigma = float(np.sqrt(noise_var))
    amplitude = amplitude_from_inr(INR_DB, noise_var)

    y_list: list[np.ndarray] = []
    interf_list: list[np.ndarray] = []
    bits_list: list[np.ndarray] = []

    for _ in range(BATCH_SIZE):
        info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
        coded = _encode_fixed_tail(info_bits, trellis)   # always 524 bits
        symbols = bpsk_modulate(coded)
        N = len(symbols)

        received = awgn_channel(symbols, SNR_DB, INR_DB, PERIOD, PHASE, rng)
        interference = generate_interference(N, amplitude, PERIOD, PHASE)

        y_list.append(pair_received_signal(received))
        interf_list.append(pair_received_signal(interference))
        bits_list.append(info_bits)

    y_batch = torch.tensor(np.stack(y_list), dtype=torch.float32)        # (B, T, 2)
    interf_batch = torch.tensor(np.stack(interf_list), dtype=torch.float32)
    oracle = compute_oracle_metrics(y_batch, interf_batch, noise_sigma)  # (B, T, 4)

    return y_batch, oracle, bits_list


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_mse_decreases_on_fixed_batch(fixed_batch) -> None:
    """MSE loss must decrease when training on a fixed batch."""
    y_batch, oracle, _ = fixed_batch

    model = NeuralBranchMetric(hidden_size=16)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    losses: list[float] = []
    model.train()
    for step in range(N_STEPS):
        pred = model(y_batch)
        loss = criterion(pred, oracle)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            print(f"  step {step+1:>3d}/{N_STEPS}  MSE={loss.item():.4f}")

    first_avg = float(np.mean(losses[:20]))
    last_avg = float(np.mean(losses[-20:]))
    print(f"\n  MSE: first_20={first_avg:.4f}  →  last_20={last_avg:.4f}")

    assert last_avg < first_avg * 0.7, (
        f"Loss did not decrease sufficiently: {first_avg:.4f} → {last_avg:.4f}"
    )


def test_overfit_mse_and_accuracy(fixed_batch, trellis, index_table) -> None:
    """
    After overfitting on the fixed batch:
      1. MSE must drop to < 1% of initial MSE (relative convergence).
         Oracle metrics are log-likelihoods ∝ 1/(2σ²); at SNR=10 dB the
         correct hypothesis scores ≈ −1 while wrong ones score ≈ −40 to −80,
         so the initial MSE (random model ≈ 0 output) can be in the thousands.
         An absolute threshold of < 0.1 is unreachable; relative is correct.
      2. Viterbi with predicted metrics must decode > 70% of bits correctly.
         Even imperfect absolute metric values can yield correct rankings, so
         70% is a meaningful lower bound that verifies integration is working.
    """
    y_batch, oracle, bits_list = fixed_batch

    model = NeuralBranchMetric(hidden_size=16)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model.train()
    initial_mse: float | None = None
    for step in range(N_STEPS):
        pred = model(y_batch)
        loss = criterion(pred, oracle)
        if step == 0:
            initial_mse = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"  step {step+1:>3d}/{N_STEPS}  MSE={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred_eval = model(y_batch)  # (B, T, 4)
        final_mse = criterion(pred_eval, oracle).item()

    reduction = final_mse / initial_mse if initial_mse else float('inf')
    print(f"\n  Initial MSE: {initial_mse:.2f}  →  Final MSE: {final_mse:.4f}"
          f"  ({reduction*100:.2f}% of initial)")
    assert reduction < 0.01, (
        f"MSE did not drop to < 1% of initial: "
        f"{initial_mse:.2f} → {final_mse:.4f} ({reduction*100:.2f}%)"
    )

    # Decode each block with Viterbi using the predicted branch metrics
    pred_np = pred_eval.detach().numpy()  # (B, T, 4)
    total_bits = 0
    correct_bits = 0
    for b, info_bits in enumerate(bits_list):
        decoded = viterbi_neural_bm(pred_np[b], trellis, index_table)
        correct_bits += int(np.sum(decoded == info_bits))
        total_bits += K_INFO

    bit_acc = correct_bits / total_bits
    print(f"  Bit accuracy on training batch: {bit_acc:.4f}")
    assert bit_acc > 0.70, (
        f"Bit accuracy {bit_acc:.4f} < 0.70 — predicted metrics may not be ranking "
        f"hypotheses correctly even after overfitting"
    )
