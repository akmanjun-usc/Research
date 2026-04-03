"""
channel.py — AWGN + sinusoidal interference channel simulator

Part of: EE597 Search-Designed Trellis Codes Project
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────
# Constants (project-wide fixed values)
# ─────────────────────────────────────────────
N_CODED = 512       # coded block length
K_INFO  = 256       # information bits per block
N_STATES = 64       # trellis states
RATE = 0.5          # code rate


# ─────────────────────────────────────────────
# BPSK modulation
# ─────────────────────────────────────────────
def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """bits in {0,1} -> symbols in {-1, +1}"""
    return 1 - 2 * bits.astype(float)


def bpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Hard decision: symbols -> bits in {0,1}"""
    return (symbols < 0).astype(np.int8)


# ─────────────────────────────────────────────
# Channel utilities
# ─────────────────────────────────────────────
def noise_var_from_snr(snr_db: float, es: float = 1.0) -> float:
    """sigma^2 = E_s / SNR_linear"""
    snr_linear = 10 ** (snr_db / 10)
    return es / snr_linear


def amplitude_from_inr(inr_db: float, noise_var: float) -> float:
    """A = sqrt(2 * INR_linear * sigma^2). Sinusoid power = A^2/2."""
    inr_linear = 10 ** (inr_db / 10)
    return np.sqrt(2 * inr_linear * noise_var)


def generate_interference(
    n_symbols: int,
    amplitude: float,
    period: float,
    phase: float,
) -> np.ndarray:
    """i[t] = A * sin(2*pi*t/P + phi)"""
    t = np.arange(n_symbols)
    return amplitude * np.sin(2 * np.pi * t / period + phase)


# ─────────────────────────────────────────────
# Full channel model
# ─────────────────────────────────────────────
def awgn_channel(
    symbols: np.ndarray,
    snr_db: float,
    inr_db: float,
    period: float,
    phase: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Full channel: r = x + n + i

    Args:
        symbols: BPSK symbols in {-1, +1}, shape (N,)
        snr_db: signal-to-noise ratio in dB
        inr_db: interference-to-noise ratio in dB
        period: sinusoidal interference period
        phase: sinusoidal interference phase
        rng: numpy random generator

    Returns:
        received signal, shape (N,)
    """
    n = len(symbols)
    noise_var = noise_var_from_snr(snr_db)
    noise = rng.standard_normal(n) * np.sqrt(noise_var)
    amplitude = amplitude_from_inr(inr_db, noise_var)
    interference = generate_interference(n, amplitude, period, phase)
    return symbols + noise + interference


def sample_channel_params(
    snr_range_db: tuple[float, float] = (0.0, 10.0),
    inr_range_db: tuple[float, float] = (-5.0, 10.0),
    period_range: tuple[int, int] = (8, 32),
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Returns dict with keys: snr_db, inr_db, period, phase, amplitude, noise_var"""
    if rng is None:
        rng = np.random.default_rng()
    snr_db = rng.uniform(snr_range_db[0], snr_range_db[1])
    inr_db = rng.uniform(inr_range_db[0], inr_range_db[1])
    period = rng.integers(period_range[0], period_range[1] + 1)
    phase = rng.uniform(0, 2 * np.pi)
    noise_var = noise_var_from_snr(snr_db)
    amplitude = amplitude_from_inr(inr_db, noise_var)
    return dict(
        snr_db=snr_db, inr_db=inr_db, period=period, phase=phase,
        amplitude=amplitude, noise_var=noise_var,
    )


# ─────────────────────────────────────────────
# Smoke test (run with: python channel.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running smoke test for channel...")
    rng = np.random.default_rng(42)

    # Test noise variance
    nv = noise_var_from_snr(0.0)
    assert abs(nv - 1.0) < 1e-10, f"noise_var at 0dB should be 1.0, got {nv}"

    nv10 = noise_var_from_snr(10.0)
    assert abs(nv10 - 0.1) < 1e-10, f"noise_var at 10dB should be 0.1, got {nv10}"

    # Test amplitude
    a = amplitude_from_inr(0.0, 1.0)
    assert abs(a - np.sqrt(2)) < 1e-10, f"amplitude at INR=0dB, nv=1 should be sqrt(2), got {a}"

    # Test interference shape
    interf = generate_interference(512, 1.0, 16.0, 0.0)
    assert interf.shape == (512,), f"interference shape wrong: {interf.shape}"
    assert abs(np.mean(interf)) < 0.1, "interference mean should be near zero"

    # Test full channel
    symbols = bpsk_modulate(rng.integers(0, 2, 512, dtype=np.int8))
    rx = awgn_channel(symbols, 5.0, 5.0, 16.0, 0.0, rng)
    assert rx.shape == (512,), f"received shape wrong: {rx.shape}"

    # Test BPSK roundtrip
    bits = rng.integers(0, 2, 100, dtype=np.int8)
    sym = bpsk_modulate(bits)
    assert np.all((sym == 1) | (sym == -1)), "BPSK symbols must be +/-1"
    bits_back = bpsk_demodulate(sym)
    assert np.array_equal(bits, bits_back), "BPSK roundtrip failed"

    # Test sample_channel_params
    params = sample_channel_params(rng=rng)
    assert all(k in params for k in ['snr_db', 'inr_db', 'period', 'phase', 'amplitude', 'noise_var'])

    print("PASS")
