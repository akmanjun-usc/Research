"""
test_encode_decode.py — Noiseless encode->decode round-trip test

100 random blocks through NASA K=7 code, decoded with mismatched Viterbi
on a noiseless channel. Must produce 0 errors on every block.
"""

import numpy as np
import pytest

from trellis import load_nasa_k7, K_INFO
from decoders import viterbi_decode, branch_metric_mismatched
from channel import bpsk_modulate


@pytest.fixture
def trellis():
    return load_nasa_k7()


def test_noiseless_roundtrip(trellis):
    """100 random blocks, noiseless encode->decode = 0 errors."""
    rng = np.random.default_rng(0)

    for trial in range(100):
        info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
        coded = trellis.encode(info_bits)
        symbols = bpsk_modulate(coded)
        received = symbols.reshape(-1, 2)
        decoded = viterbi_decode(
            received, trellis, branch_metric_mismatched, noise_var=1.0,
        )
        assert np.array_equal(info_bits, decoded), (
            f"Trial {trial}: noiseless round-trip FAILED. "
            f"First diff at bit {np.where(info_bits != decoded)[0][0]}"
        )


def test_allzero_roundtrip(trellis):
    """All-zero input should produce all-zero output."""
    info_bits = np.zeros(K_INFO, dtype=np.int8)
    coded = trellis.encode(info_bits)
    symbols = bpsk_modulate(coded)
    received = symbols.reshape(-1, 2)
    decoded = viterbi_decode(
        received, trellis, branch_metric_mismatched, noise_var=1.0,
    )
    assert np.array_equal(info_bits, decoded), "All-zero round-trip FAILED"


def test_encode_output_length(trellis):
    """Encoded output should have the correct length."""
    rng = np.random.default_rng(42)
    info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
    coded = trellis.encode(info_bits)
    # K=7 code: K_INFO info bits + 6 tail bits, rate 1/2
    expected_len = 2 * (K_INFO + 6)
    assert len(coded) == expected_len, (
        f"Expected {expected_len} coded bits, got {len(coded)}"
    )


def test_encode_binary_output(trellis):
    """All coded bits must be 0 or 1."""
    rng = np.random.default_rng(42)
    info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
    coded = trellis.encode(info_bits)
    assert np.all((coded == 0) | (coded == 1)), "Coded bits must be binary"
