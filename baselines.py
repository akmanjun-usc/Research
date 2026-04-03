"""
baselines.py — Wrapper that ties B1/B2/B5 baselines together

Part of: EE597 Search-Designed Trellis Codes Project
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from pathlib import Path

from channel import (
    bpsk_modulate, awgn_channel, noise_var_from_snr,
    amplitude_from_inr, generate_interference, K_INFO,
)
from trellis import load_nasa_k7, Trellis
from decoders import viterbi_decode, branch_metric_mismatched, branch_metric_oracle
from interference_est import estimate_and_cancel


# ─────────────────────────────────────────────
# Baseline runner
# ─────────────────────────────────────────────
def run_baseline_b1(
    info_bits: np.ndarray,
    trellis: Trellis,
    received: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """B1: Mismatched Viterbi — ignores interference."""
    nv = noise_var_from_snr(snr_db)
    rx = received.reshape(-1, 2)
    return viterbi_decode(rx, trellis, branch_metric_mismatched, noise_var=nv)


def run_baseline_b2(
    info_bits: np.ndarray,
    trellis: Trellis,
    received: np.ndarray,
    snr_db: float,
    inr_db: float,
    period: float,
    phase: float,
) -> np.ndarray:
    """B2: Oracle Viterbi — knows interference perfectly."""
    nv = noise_var_from_snr(snr_db)
    amp = amplitude_from_inr(inr_db, nv)
    interf = generate_interference(len(received), amp, period, phase)
    rx = received.reshape(-1, 2)
    return viterbi_decode(rx, trellis, branch_metric_oracle,
                          noise_var=nv, interference=interf)


def run_baseline_b5(
    info_bits: np.ndarray,
    trellis: Trellis,
    received: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """B5: Interference cancellation + mismatched Viterbi."""
    nv = noise_var_from_snr(snr_db)
    cleaned, _ = estimate_and_cancel(received, period_range=(8, 32))
    rx = cleaned.reshape(-1, 2)
    return viterbi_decode(rx, trellis, branch_metric_mismatched, noise_var=nv)


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running smoke test for baselines...")
    rng = np.random.default_rng(42)

    trellis = load_nasa_k7()
    info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
    coded = trellis.encode(info_bits)
    symbols = bpsk_modulate(coded)

    # Add channel
    rx = awgn_channel(symbols, snr_db=8.0, inr_db=5.0, period=16.0, phase=0.5, rng=rng)

    # B1
    dec_b1 = run_baseline_b1(info_bits, trellis, rx, 8.0)
    err_b1 = np.sum(info_bits != dec_b1)
    print(f"  B1 (mismatched): {err_b1} bit errors")

    # B2
    dec_b2 = run_baseline_b2(info_bits, trellis, rx, 8.0, 5.0, 16.0, 0.5)
    err_b2 = np.sum(info_bits != dec_b2)
    print(f"  B2 (oracle): {err_b2} bit errors")

    # B5
    dec_b5 = run_baseline_b5(info_bits, trellis, rx, 8.0)
    err_b5 = np.sum(info_bits != dec_b5)
    print(f"  B5 (IC+Viterbi): {err_b5} bit errors")

    print("PASS")
