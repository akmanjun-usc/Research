"""
test_nbm_vs_b1.py — Verify N2 (neural branch metric) beats B1 under interference

At SNR=5 dB, INR=5 dB, sinusoidal interference with P ∈ [8,32]:
  B1 (mismatched Viterbi) ignores the interference → degraded BLER
  N2 (neural branch metric + Viterbi) learns interference-aware metrics → lower BLER

Success criterion (from CLAUDE.md §2b):
  N2 BLER < B1 BLER at SNR=5 dB, INR=5 dB
  (≥ 0.5 dB gap is the stretch goal; this test verifies the direction)

This test requires a fully trained N2 model.  It is skipped if no checkpoint
exists at results/phase2b/checkpoints/best_model_seed42.pt.  Run
  python neural_bm.py --train
to produce the checkpoint first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_bm import (
    build_branch_output_index,
    load_model,
    make_decoder_n2,
)
from eval import estimate_bler, make_encoder, make_decoder_b1
from channel import awgn_channel
from trellis import load_nasa_k7

CHECKPOINT_PATH = Path("results/phase2b/checkpoints/best_model_seed42.pt")

EVAL_SNR_DB = 5.0
EVAL_INR_DB = 5.0
N_TRIALS = 1000
SEED = 77


def _requires_checkpoint():
    return pytest.mark.skipif(
        not CHECKPOINT_PATH.exists(),
        reason=(
            f"No trained checkpoint at {CHECKPOINT_PATH}. "
            "Run:  python neural_bm.py --train --seed 42"
        ),
    )


@pytest.fixture(scope="module")
def trellis():
    return load_nasa_k7()


@pytest.fixture(scope="module")
def encoder(trellis):
    return make_encoder(trellis)


@pytest.fixture(scope="module")
def n2_model():
    model, _ = load_model(CHECKPOINT_PATH)
    return model


@_requires_checkpoint()
@pytest.mark.slow
def test_n2_beats_b1_under_interference(trellis, encoder, n2_model) -> None:
    """
    N2 BLER must be lower than B1 BLER at SNR=5 dB, INR=5 dB.

    B1 ignores the sinusoidal interference entirely, suffering metric mismatch.
    N2 learns interference-aware branch metrics, enabling Viterbi to search
    more accurately despite the interference.
    """
    index_table = build_branch_output_index(trellis)
    decode_n2 = make_decoder_n2(n2_model, 'cpu', trellis, index_table)
    decode_b1 = make_decoder_b1(trellis)

    r_n2 = estimate_bler(
        encoder, decode_n2, awgn_channel,
        snr_db=EVAL_SNR_DB, inr_db=EVAL_INR_DB,
        n_trials=N_TRIALS, seed=SEED,
    )
    r_b1 = estimate_bler(
        encoder, decode_b1, awgn_channel,
        snr_db=EVAL_SNR_DB, inr_db=EVAL_INR_DB,
        n_trials=N_TRIALS, seed=SEED,
    )

    print(f"\n  N2 BLER @ SNR={EVAL_SNR_DB}dB, INR={EVAL_INR_DB}dB: "
          f"{r_n2['bler']:.4f}  ({r_n2['n_errors']}/{r_n2['n_trials']})")
    print(f"  B1 BLER @ SNR={EVAL_SNR_DB}dB, INR={EVAL_INR_DB}dB: "
          f"{r_b1['bler']:.4f}  ({r_b1['n_errors']}/{r_b1['n_trials']})")

    if r_b1['bler'] > 0 and r_n2['bler'] > 0:
        # Compute approximate dB gain at BLER operating point
        # (positive = N2 better)
        ratio = r_b1['bler'] / r_n2['bler']
        print(f"  BLER ratio B1/N2: {ratio:.2f}×")

    assert r_n2['bler'] < r_b1['bler'], (
        f"N2 BLER ({r_n2['bler']:.4f}) is not lower than B1 BLER ({r_b1['bler']:.4f}) "
        f"under sinusoidal interference — model may need more training"
    )


@_requires_checkpoint()
@pytest.mark.slow
def test_n2_bler_improves_with_snr(trellis, encoder, n2_model) -> None:
    """N2 BLER should decrease as SNR increases at fixed INR=5 dB."""
    index_table = build_branch_output_index(trellis)
    decode_n2 = make_decoder_n2(n2_model, 'cpu', trellis, index_table)

    blers: list[float] = []
    for snr_db in [3.0, 5.0, 7.0]:
        r = estimate_bler(
            encoder, decode_n2, awgn_channel,
            snr_db=snr_db, inr_db=EVAL_INR_DB,
            n_trials=500, seed=SEED,
        )
        blers.append(r['bler'])
        print(f"  N2 BLER @ SNR={snr_db}dB, INR={EVAL_INR_DB}dB: {r['bler']:.4f}")

    # BLER should not increase as SNR increases (allow small statistical noise)
    for i in range(1, len(blers)):
        assert blers[i] <= blers[i - 1] + 0.05, (
            f"N2 BLER not decreasing with SNR: {blers}"
        )
