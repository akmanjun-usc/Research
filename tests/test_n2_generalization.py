"""
test_n2_generalization.py — Verify N2 generalizes to unseen random trellises

Phase 3b uses the frozen N2 model as a fitness signal for EA search over
arbitrary 64-state trellises.  This test checks that N2's branch metrics
remain informative (better than random) when the trellis changes from NASA K=7.

Success criterion: median BLER < 0.5 across 10 random valid trellises at
SNR=5 dB, INR=5 dB.  BLER < 0.5 means N2 + Viterbi beats random guessing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from channel import awgn_channel
from eval import estimate_bler, make_encoder
from neural_bm import build_branch_output_index, load_model, make_decoder_n2
from trellis_genome import genome_to_trellis, random_valid_genome

CHECKPOINT = Path("results/phase2b/checkpoints/best_model_seed32.pt")
N_TRELLISES = 10
N_TRIALS = 1000
SNR_DB = 5.0
INR_DB = 5.0
SEED = 42


@pytest.fixture(scope="module")
def n2_model():
    if not CHECKPOINT.exists():
        pytest.skip(f"No checkpoint at {CHECKPOINT}")
    model, _ = load_model(CHECKPOINT)
    return model


@pytest.mark.skipif(not CHECKPOINT.exists(), reason=f"No checkpoint at {CHECKPOINT}")
@pytest.mark.slow
def test_n2_generalizes_to_random_trellises(n2_model) -> None:
    rng = np.random.default_rng(SEED)
    blers: list[float] = []

    for i in range(N_TRELLISES):
        genome = random_valid_genome(rng)
        trellis = genome_to_trellis(genome)
        encode_fn = make_encoder(trellis)
        index_table = build_branch_output_index(trellis)
        decode_fn = make_decoder_n2(n2_model, "cpu", trellis, index_table)

        result = estimate_bler(
            encode_fn,
            decode_fn,
            awgn_channel,
            snr_db=SNR_DB,
            inr_db=INR_DB,
            n_trials=N_TRIALS,
            seed=SEED + i,
        )
        bler = result["bler"]
        blers.append(bler)
        print(
            f"  trellis {i:2d}: BLER={bler:.4f}"
            f"  ({result['n_errors']}/{result['n_trials']} errors)"
        )

    median_bler = float(np.median(blers))
    print(f"\n  Median BLER across {N_TRELLISES} random trellises: {median_bler:.4f}")

    assert median_bler < 0.5, (
        f"N2 median BLER ({median_bler:.4f}) >= 0.5 on random trellises — "
        "N2 does not generalize; Phase 3b fitness signal is unreliable."
    )
