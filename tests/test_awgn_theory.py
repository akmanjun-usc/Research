"""
test_awgn_theory.py — Validate AWGN-only performance vs theory

Run K=7 code with mismatched Viterbi on AWGN-only channel (no interference).
BLER should be reasonable for a K=7 code — at high SNR, errors should be rare.
"""

import numpy as np
import pytest

from trellis import load_nasa_k7, K_INFO
from decoders import viterbi_decode, branch_metric_mismatched
from channel import bpsk_modulate, noise_var_from_snr


@pytest.fixture
def trellis():
    return load_nasa_k7()


def run_awgn_bler(trellis, snr_db: float, n_trials: int, seed: int = 0) -> dict:
    """Run BLER estimation on AWGN-only channel (no interference)."""
    rng = np.random.default_rng(seed)
    n_errors = 0
    noise_var = noise_var_from_snr(snr_db)

    for _ in range(n_trials):
        info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
        coded = trellis.encode(info_bits)
        symbols = bpsk_modulate(coded)
        noise = rng.standard_normal(len(symbols)) * np.sqrt(noise_var)
        received = (symbols + noise).reshape(-1, 2)
        decoded = viterbi_decode(
            received, trellis, branch_metric_mismatched, noise_var=noise_var,
        )
        if not np.array_equal(info_bits, decoded):
            n_errors += 1

    bler = n_errors / n_trials
    return dict(bler=bler, n_errors=n_errors, n_trials=n_trials, snr_db=snr_db)


def test_awgn_high_snr_low_bler(trellis):
    """At SNR=8 dB, K=7 code should have very low BLER on AWGN."""
    result = run_awgn_bler(trellis, snr_db=8.0, n_trials=1000, seed=42)
    print(f"  AWGN SNR=8dB: BLER={result['bler']:.4f} ({result['n_errors']}/{result['n_trials']})")
    # K=7 d_free=10 code should have BLER < 0.01 at 8 dB
    assert result['bler'] < 0.05, (
        f"BLER at 8 dB too high: {result['bler']:.4f}. "
        f"K=7 code should perform well at this SNR."
    )


def test_awgn_bler_decreases_with_snr(trellis):
    """BLER should monotonically decrease (roughly) as SNR increases."""
    snr_points = [2.0, 4.0, 6.0, 8.0]
    bler_values = []

    for snr_db in snr_points:
        result = run_awgn_bler(trellis, snr_db=snr_db, n_trials=2000, seed=42)
        bler_values.append(result['bler'])
        print(f"  AWGN SNR={snr_db:.0f}dB: BLER={result['bler']:.4f}")

    # Check that BLER at highest SNR is less than at lowest SNR
    assert bler_values[-1] < bler_values[0], (
        f"BLER should decrease with SNR. Got {bler_values}"
    )

    # Check that high-SNR BLER is substantially lower
    assert bler_values[-1] < 0.5 * bler_values[0], (
        f"BLER drop too small: {bler_values[0]:.4f} -> {bler_values[-1]:.4f}"
    )


def test_awgn_low_snr_high_bler(trellis):
    """At very low SNR (0 dB), BLER should be high (near 1)."""
    result = run_awgn_bler(trellis, snr_db=0.0, n_trials=500, seed=42)
    print(f"  AWGN SNR=0dB: BLER={result['bler']:.4f}")
    assert result['bler'] > 0.5, (
        f"BLER at 0 dB unexpectedly low: {result['bler']:.4f}"
    )
