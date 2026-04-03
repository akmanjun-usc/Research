"""
interference_est.py — Interference estimation + cancellation (Baseline B5)

Part of: EE597 Search-Designed Trellis Codes Project

Approach:
1. FFT/periodogram to detect dominant frequency -> estimate period P
2. Least-squares fit for amplitude A and phase phi
3. Subtract estimated interference: y_clean[t] = r[t] - A_hat * sin(2*pi*t/P_hat + phi_hat)
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
N_CODED = 512
K_INFO  = 256


# ─────────────────────────────────────────────
# Interference estimation
# ─────────────────────────────────────────────
def estimate_interference(
    received: np.ndarray,
    period_range: tuple[int, int] = (8, 32),
    n_candidates: int = 5,
) -> dict:
    """
    Estimate sinusoidal interference parameters from received signal.

    Uses FFT to find dominant frequency, then least-squares to refine
    amplitude and phase.

    Args:
        received: shape (N,) — received signal
        period_range: valid period range to search
        n_candidates: number of frequency candidates to try

    Returns:
        dict with keys: amplitude, period, phase
    """
    N = len(received)

    # Step 1: FFT to find dominant frequency
    fft_vals = np.fft.rfft(received)
    freqs = np.fft.rfftfreq(N)
    magnitudes = np.abs(fft_vals)

    # Only consider frequencies corresponding to periods in valid range
    # period = 1/freq, so freq in [1/P_max, 1/P_min]
    freq_min = 1.0 / period_range[1]
    freq_max = 1.0 / period_range[0]

    valid_mask = (freqs >= freq_min) & (freqs <= freq_max)
    if not np.any(valid_mask):
        # No valid frequency found — return zero interference estimate
        return dict(amplitude=0.0, period=float(period_range[0]), phase=0.0)

    valid_indices = np.where(valid_mask)[0]
    valid_mags = magnitudes[valid_indices]

    # Get top candidates
    top_k = min(n_candidates, len(valid_indices))
    top_idx = valid_indices[np.argsort(valid_mags)[-top_k:]]

    # Step 2: For each candidate, fit amplitude and phase via least-squares
    t = np.arange(N)
    best_residual = np.inf
    best_params = dict(amplitude=0.0, period=float(period_range[0]), phase=0.0)

    for idx in top_idx:
        freq = freqs[idx]
        if freq == 0:
            continue
        period = 1.0 / freq

        # Fit: r[t] ≈ a*cos(2*pi*t/P) + b*sin(2*pi*t/P) + DC
        # Using least squares: [cos, sin, 1] @ [a, b, c] ≈ r
        omega = 2 * np.pi / period
        A_mat = np.column_stack([
            np.cos(omega * t),
            np.sin(omega * t),
            np.ones(N),
        ])
        coeffs, residual_arr, _, _ = np.linalg.lstsq(A_mat, received, rcond=None)
        a_cos, a_sin, dc = coeffs

        # Convert to amplitude and phase: A*sin(wt + phi) = A*sin(phi)*cos(wt) + A*cos(phi)*sin(wt)
        # So a_cos = A*sin(phi), a_sin = A*cos(phi)
        amplitude = np.sqrt(a_cos**2 + a_sin**2)
        phase = np.arctan2(a_cos, a_sin)

        # Compute residual
        estimated = amplitude * np.sin(omega * t + phase)
        residual = np.sum((received - estimated)**2)

        if residual < best_residual:
            best_residual = residual
            best_params = dict(amplitude=amplitude, period=period, phase=phase)

    return best_params


def cancel_interference(
    received: np.ndarray,
    amplitude: float,
    period: float,
    phase: float,
) -> np.ndarray:
    """
    Subtract estimated interference from received signal.

    Args:
        received: shape (N,) — received signal
        amplitude: estimated interference amplitude
        period: estimated interference period
        phase: estimated interference phase

    Returns:
        cleaned signal, shape (N,)
    """
    t = np.arange(len(received))
    estimated_interference = amplitude * np.sin(2 * np.pi * t / period + phase)
    return received - estimated_interference


def estimate_and_cancel(
    received: np.ndarray,
    period_range: tuple[int, int] = (8, 32),
) -> tuple[np.ndarray, dict]:
    """
    Combined estimation + cancellation pipeline.

    Returns:
        (cleaned_signal, estimated_params)
    """
    params = estimate_interference(received, period_range=period_range)
    cleaned = cancel_interference(
        received, params['amplitude'], params['period'], params['phase'],
    )
    return cleaned, params


# ─────────────────────────────────────────────
# Smoke test (run with: python interference_est.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running smoke test for interference_est...")
    rng = np.random.default_rng(42)

    # Generate known interference + noise
    N = 512
    true_amp = 1.5
    true_period = 16.0
    true_phase = 0.7

    from channel import generate_interference
    interf = generate_interference(N, true_amp, true_period, true_phase)
    noise = rng.standard_normal(N) * 0.3
    # Simulate received = BPSK + interference + noise
    bpsk_signal = 2 * rng.integers(0, 2, N) - 1.0
    received = bpsk_signal + interf + noise

    # Estimate
    params = estimate_interference(received, period_range=(8, 32))
    print(f"  True:  A={true_amp:.3f}, P={true_period:.1f}, phi={true_phase:.3f}")
    print(f"  Est:   A={params['amplitude']:.3f}, P={params['period']:.1f}, phi={params['phase']:.3f}")

    # Check estimation quality
    amp_err = abs(params['amplitude'] - true_amp) / true_amp
    period_err = abs(params['period'] - true_period) / true_period
    print(f"  Amp error: {amp_err*100:.1f}%, Period error: {period_err*100:.1f}%")

    # Cancel
    cleaned, _ = estimate_and_cancel(received, period_range=(8, 32))
    # Residual interference power should be much less than original
    orig_power = np.mean(interf**2)
    resid = cleaned - bpsk_signal - noise  # ideally zero
    resid_power = np.mean(resid**2)
    cancellation_db = 10 * np.log10(orig_power / max(resid_power, 1e-20))
    print(f"  Cancellation: {cancellation_db:.1f} dB")

    assert cancellation_db > 5.0, f"Poor cancellation: only {cancellation_db:.1f} dB"
    print("PASS")
