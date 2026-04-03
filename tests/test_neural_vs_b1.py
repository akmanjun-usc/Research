"""
test_neural_vs_b1.py — Verify trained N1 (BiGRU) beats B1 (mismatched Viterbi)

This test requires a trained checkpoint. Mark as slow since it runs
Monte Carlo evaluation.
"""

import pytest
import numpy as np
from pathlib import Path

from neural_decoder import load_model, make_decoder_n1, RESULTS_DIR
from eval import estimate_bler, make_encoder, make_decoder_b1
from channel import awgn_channel, K_INFO
from trellis import load_nasa_k7


CHECKPOINT_PATH = RESULTS_DIR / "checkpoints" / "best_gru.pt"


@pytest.fixture(scope="module")
def trellis():
    return load_nasa_k7()


@pytest.fixture(scope="module")
def encoder(trellis):
    return make_encoder(trellis)


def requires_checkpoint(fn):
    """Skip test if no trained checkpoint exists."""
    return pytest.mark.skipif(
        not CHECKPOINT_PATH.exists(),
        reason=f"No checkpoint at {CHECKPOINT_PATH} — train first",
    )(fn)


@requires_checkpoint
@pytest.mark.slow
def test_n1_beats_b1_at_5db(trellis, encoder):
    """N1 (BiGRU) should have lower BLER than B1 at SNR=5dB, INR=5dB."""
    model = load_model(CHECKPOINT_PATH, device="cpu")
    decode_n1 = make_decoder_n1(model, device="cpu")
    decode_b1 = make_decoder_b1(trellis)

    n_trials = 5000
    snr_db, inr_db = 5.0, 5.0

    r_n1 = estimate_bler(encoder, decode_n1, awgn_channel,
                         snr_db=snr_db, inr_db=inr_db,
                         n_trials=n_trials, seed=123)
    r_b1 = estimate_bler(encoder, decode_b1, awgn_channel,
                         snr_db=snr_db, inr_db=inr_db,
                         n_trials=n_trials, seed=123)

    print(f"N1 BLER @ SNR={snr_db}: {r_n1['bler']:.4f} "
          f"({r_n1['n_errors']}/{r_n1['n_trials']})")
    print(f"B1 BLER @ SNR={snr_db}: {r_b1['bler']:.4f} "
          f"({r_b1['n_errors']}/{r_b1['n_trials']})")

    assert r_n1['bler'] < r_b1['bler'], \
        f"N1 ({r_n1['bler']:.4f}) should beat B1 ({r_b1['bler']:.4f})"


@requires_checkpoint
@pytest.mark.slow
def test_n1_bler_decreases_with_snr(trellis, encoder):
    """N1 BLER should decrease as SNR increases."""
    model = load_model(CHECKPOINT_PATH, device="cpu")
    decode_n1 = make_decoder_n1(model, device="cpu")

    blers = []
    for snr_db in [3.0, 5.0, 7.0]:
        r = estimate_bler(encoder, decode_n1, awgn_channel,
                          snr_db=snr_db, inr_db=5.0,
                          n_trials=2000, seed=456)
        blers.append(r['bler'])
        print(f"N1 BLER @ SNR={snr_db}: {r['bler']:.4f}")

    # Each BLER should be <= the previous (higher SNR = lower BLER)
    for i in range(1, len(blers)):
        assert blers[i] <= blers[i - 1] + 0.02, \
            f"BLER not decreasing: {blers}"
