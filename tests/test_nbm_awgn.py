"""
test_nbm_awgn.py — Verify N2 ≈ B1 on pure AWGN (no interference)

On AWGN-only channel the oracle branch metric reduces to the standard
mismatched Euclidean metric (interference = 0, so all 4 hypotheses compete
solely on noise).  A trained N2 should learn to approximate this and
achieve BLER close to B1 (which is optimal on AWGN).

Test strategy:
  - Train a small model on AWGN-only data (INR = −100 dB ≈ 0 interference).
  - Evaluate BLER at SNR = 5 dB with no interference.
  - Assert N2 BLER ≤ 2× B1 BLER (parity check; not a performance gain test).

This test requires a non-trivial training run so it is marked @pytest.mark.slow.
Skip if a pre-trained AWGN checkpoint exists to avoid re-training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_bm import (
    DEFAULT_CONFIG,
    NeuralBranchMetric,
    build_branch_output_index,
    load_model,
    make_decoder_n2,
    train_neural_bm,
)
from eval import estimate_bler, make_encoder, make_decoder_b1
from channel import awgn_channel
from trellis import load_nasa_k7

CHECKPOINT_PATH = Path("results/phase2b/checkpoints/awgn_test_model.pt")

AWGN_TRAIN_CONFIG = {
    **DEFAULT_CONFIG,
    'hidden_size': 16,
    'batch_size': 64,
    'batches_per_epoch': 5,
    'num_epochs': 100,
    'patience': 10,
    'val_every': 5,
    'val_blocks': 200,
    'snr_range': (3.0, 7.0),
    'inr_range': (-100.0, -100.0),   # pure AWGN (amplitude ≈ 0)
    'val_snr': 5.0,
    'val_inr': -100.0,
    'checkpoint_dir': 'results/phase2b/checkpoints',
}

EVAL_SNR_DB = 5.0
EVAL_INR_DB = -100.0   # pure AWGN
N_TRIALS = 500
SEED = 99


@pytest.fixture(scope="module")
def trellis():
    return load_nasa_k7()


@pytest.fixture(scope="module")
def encoder(trellis):
    return make_encoder(trellis)


@pytest.fixture(scope="module")
def awgn_model(trellis):
    """Return trained or load cached AWGN-only model."""
    if CHECKPOINT_PATH.exists():
        print(f"Loading pre-trained AWGN model from {CHECKPOINT_PATH}...")
        model, _ = load_model(CHECKPOINT_PATH)
        return model
    # Train a small model quickly
    cfg = {**AWGN_TRAIN_CONFIG, 'checkpoint_dir': str(CHECKPOINT_PATH.parent)}
    model = train_neural_bm(cfg, seed=SEED)
    return model


@pytest.mark.slow
def test_n2_awgn_parity_with_b1(trellis, encoder, awgn_model) -> None:
    """
    N2 trained on AWGN-only should achieve BLER within 2× of B1 at SNR=5 dB.

    On pure AWGN the oracle metric = mismatched metric, so N2 should learn
    approximately the same branch metric as B1 (standard Euclidean distance).
    """
    index_table = build_branch_output_index(trellis)
    decode_n2 = make_decoder_n2(awgn_model, 'cpu', trellis, index_table)
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

    print(f"\n  N2 BLER (AWGN only, SNR={EVAL_SNR_DB}dB): {r_n2['bler']:.4f}")
    print(f"  B1 BLER (AWGN only, SNR={EVAL_SNR_DB}dB): {r_b1['bler']:.4f}")

    # N2 should not be catastrophically worse than B1 on clean AWGN
    # Allow up to 2× overhead (generous margin for limited training)
    if r_b1['bler'] > 0:
        ratio = r_n2['bler'] / r_b1['bler']
        print(f"  N2/B1 BLER ratio: {ratio:.2f}×")
        assert ratio <= 2.0, (
            f"N2 BLER ({r_n2['bler']:.4f}) is more than 2× B1 BLER ({r_b1['bler']:.4f}) "
            f"on pure AWGN — the NN may not have converged"
        )
    else:
        # B1 has zero errors at this SNR — N2 should too
        assert r_n2['bler'] <= 0.05, (
            f"N2 BLER {r_n2['bler']:.4f} > 0.05 when B1 achieves 0.0 on pure AWGN"
        )
