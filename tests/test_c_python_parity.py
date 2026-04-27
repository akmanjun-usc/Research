import numpy as np
import pytest
from functools import lru_cache

from constraints import compute_dfree, is_fully_connected, is_non_catastrophic, is_terminating
from neural_bm import _encode_fixed_tail, build_branch_output_index, viterbi_neural_bm
from phase3_native import (
    check_connectivity_native,
    check_noncatastrophic_native,
    check_termination_native,
    compute_dfree_native,
    encode_native,
    is_available,
    mutate_and_validate_native,
    viterbi_neural_bm_native,
)
from trellis import Trellis, load_nasa_k7
from trellis_genome import BITS_FROM_PAIR, genome_to_trellis, nasa_k7_genome, random_valid_genome


def _catastrophic_trellis() -> Trellis:
    next_state = np.array([[0, 1], [0, 0], [3, 0], [2, 1]], dtype=np.int32)
    output_bits = np.array(
        [[[0, 0], [0, 0]], [[0, 0], [1, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]],
        dtype=np.int8,
    )
    return Trellis(n_states=4, next_state=next_state, output_bits=output_bits)


pytestmark = pytest.mark.skipif(not is_available(), reason="phase3_core.so not built")


@lru_cache(maxsize=1)
def _valid_trellises():
    rng = np.random.default_rng(7)
    trellises = [load_nasa_k7()]
    for _ in range(50):
        genome = random_valid_genome(rng, max_tries=100_000)
        assert genome is not None
        trellises.append(genome_to_trellis(genome))
    return trellises


def test_encode_parity():
    trellis = load_nasa_k7()
    rng = np.random.default_rng(11)
    for _ in range(100):
        info_bits = rng.integers(0, 2, size=256, dtype=np.int8)
        py = _encode_fixed_tail(info_bits, trellis)
        c = encode_native(trellis.next_state, trellis.output_bits, info_bits, trellis.n_states, 256, 6)
        assert np.array_equal(c, py)


def test_viterbi_neural_bm_parity():
    trellis = load_nasa_k7()
    index_table = build_branch_output_index(trellis)
    rng = np.random.default_rng(12)
    for _ in range(100):
        branch_metrics = rng.standard_normal((262, 4))
        py = viterbi_neural_bm(branch_metrics, trellis, index_table)
        c = viterbi_neural_bm_native(branch_metrics, trellis.next_state, index_table, 64, 262, 256)
        assert np.array_equal(c, py)


def test_connectivity_parity():
    for trellis in _valid_trellises():
        assert check_connectivity_native(trellis.next_state, trellis.n_states) == is_fully_connected(trellis)


def test_termination_parity():
    for trellis in _valid_trellises():
        assert check_termination_native(trellis.next_state, trellis.n_states, 6) == is_terminating(trellis)


def test_noncatastrophic_parity():
    for trellis in _valid_trellises() + [_catastrophic_trellis()]:
        assert check_noncatastrophic_native(trellis.next_state, trellis.output_bits, trellis.n_states) == is_non_catastrophic(trellis)


def test_dfree_parity():
    for trellis in _valid_trellises() + [_catastrophic_trellis()]:
        assert compute_dfree_native(trellis.next_state, trellis.output_bits, trellis.n_states) == compute_dfree(trellis)


def test_mutate_validate_child_valid():
    result = mutate_and_validate_native(nasa_k7_genome(), n_edges=2, max_attempts=1000, dfree_target=8, seed=1234)
    assert result is not None
    child, dfree, _ = result
    trellis = genome_to_trellis(child)
    assert is_fully_connected(trellis)
    assert is_terminating(trellis)
    assert is_non_catastrophic(trellis)
    assert compute_dfree(trellis) == dfree
