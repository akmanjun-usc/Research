"""
test_oracle_vs_mismatch.py — Oracle Viterbi should always be >= mismatched Viterbi

The oracle decoder knows the interference perfectly and should never perform
worse than the mismatched decoder that ignores it.
"""

import numpy as np
import pytest

from trellis import load_nasa_k7, K_INFO
from decoders import viterbi_decode, branch_metric_mismatched, branch_metric_oracle
from channel import (
    bpsk_modulate, awgn_channel, noise_var_from_snr,
    amplitude_from_inr, generate_interference,
)


@pytest.fixture
def trellis():
    return load_nasa_k7()


def run_ber_comparison(
    trellis,
    snr_db: float,
    inr_db: float,
    n_trials: int,
    seed: int = 0,
) -> tuple[int, int]:
    """
    Run n_trials and count bit errors for both mismatched and oracle decoders.

    Returns:
        (mismatched_errors, oracle_errors)
    """
    rng = np.random.default_rng(seed)
    mm_errors = 0
    or_errors = 0

    noise_var = noise_var_from_snr(snr_db)
    amplitude = amplitude_from_inr(inr_db, noise_var)

    for _ in range(n_trials):
        info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
        coded = trellis.encode(info_bits)
        symbols = bpsk_modulate(coded)

        period = rng.integers(8, 33)
        phase = rng.uniform(0, 2 * np.pi)

        received = awgn_channel(symbols, snr_db, inr_db, period, phase, rng)
        rx = received.reshape(-1, 2)
        interference = generate_interference(len(symbols), amplitude, period, phase)

        # Mismatched decode
        dec_mm = viterbi_decode(rx, trellis, branch_metric_mismatched, noise_var=noise_var)
        mm_errors += np.sum(info_bits != dec_mm)

        # Oracle decode
        dec_or = viterbi_decode(rx, trellis, branch_metric_oracle,
                                noise_var=noise_var, interference=interference)
        or_errors += np.sum(info_bits != dec_or)

    return mm_errors, or_errors


def test_oracle_beats_mismatched_moderate_inr(trellis):
    """Oracle BER <= mismatched BER at INR=5dB across SNR range."""
    for snr_db in [3.0, 5.0, 7.0]:
        mm_err, or_err = run_ber_comparison(
            trellis, snr_db=snr_db, inr_db=5.0, n_trials=500, seed=42,
        )
        print(f"  SNR={snr_db}dB, INR=5dB: mismatch={mm_err}, oracle={or_err}")
        assert or_err <= mm_err, (
            f"Oracle worse than mismatched at SNR={snr_db}dB: "
            f"oracle={or_err} > mismatch={mm_err}"
        )


def test_oracle_beats_mismatched_high_inr(trellis):
    """At high INR (10 dB), oracle advantage should be large."""
    mm_err, or_err = run_ber_comparison(
        trellis, snr_db=5.0, inr_db=10.0, n_trials=500, seed=42,
    )
    print(f"  SNR=5dB, INR=10dB: mismatch={mm_err}, oracle={or_err}")
    assert or_err <= mm_err, (
        f"Oracle worse than mismatched at high INR: "
        f"oracle={or_err} > mismatch={mm_err}"
    )
    # At high INR, oracle should be significantly better
    if mm_err > 0:
        improvement = (mm_err - or_err) / mm_err
        print(f"  Oracle improvement: {improvement*100:.1f}%")
        assert improvement > 0.1, (
            f"Oracle improvement too small at high INR: {improvement*100:.1f}%"
        )


def test_oracle_equals_mismatched_no_interference(trellis):
    """With no interference (INR=-100 dB), both should give same result."""
    rng = np.random.default_rng(42)
    noise_var = noise_var_from_snr(5.0)

    n_same = 0
    n_trials = 50
    for _ in range(n_trials):
        info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
        coded = trellis.encode(info_bits)
        symbols = bpsk_modulate(coded)

        # AWGN only (effectively no interference)
        noise = rng.standard_normal(len(symbols)) * np.sqrt(noise_var)
        received = symbols + noise
        rx = received.reshape(-1, 2)

        # Zero interference
        interference = np.zeros(len(symbols))

        dec_mm = viterbi_decode(rx, trellis, branch_metric_mismatched, noise_var=noise_var)
        dec_or = viterbi_decode(rx, trellis, branch_metric_oracle,
                                noise_var=noise_var, interference=interference)

        if np.array_equal(dec_mm, dec_or):
            n_same += 1

    print(f"  No interference: {n_same}/{n_trials} identical decodings")
    assert n_same == n_trials, (
        f"With zero interference, oracle and mismatched should give identical results. "
        f"Only {n_same}/{n_trials} matched."
    )
