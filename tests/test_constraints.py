import numpy as np

from constraints import (
    compute_dfree,
    is_fully_connected,
    is_non_catastrophic,
    is_terminating,
)
from trellis import Trellis, load_nasa_k7


def _catastrophic_trellis() -> Trellis:
    next_state = np.array(
        [
            [0, 1],
            [0, 0],
            [3, 0],
            [2, 1],
        ],
        dtype=np.int32,
    )
    output_bits = np.array(
        [
            [[0, 0], [0, 0]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]],
        ],
        dtype=np.int8,
    )
    return Trellis(n_states=4, next_state=next_state, output_bits=output_bits, name="catastrophic")


def test_dfree_nasa_k7():
    assert compute_dfree(load_nasa_k7()) == 10.0


def test_non_catastrophic_nasa_k7():
    assert is_non_catastrophic(load_nasa_k7()) is True


def test_fully_connected_nasa_k7():
    assert is_fully_connected(load_nasa_k7()) is True


def test_terminating_nasa_k7():
    assert is_terminating(load_nasa_k7()) is True


def test_catastrophic_code_dfree():
    trellis = _catastrophic_trellis()
    assert is_non_catastrophic(trellis) is False
    assert compute_dfree(trellis) == float("inf")
