"""
phase3_native.py — ctypes wrappers for Phase 3 native helpers
"""

from __future__ import annotations

import ctypes
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from constraints import compute_dfree, is_fully_connected, is_non_catastrophic, is_terminating
from neural_bm import viterbi_neural_bm
from trellis import Trellis
from trellis_genome import BITS_FROM_PAIR, TrellisGenome, genome_to_trellis, mutate_and_validate

_C_LIB = None
_C_LOAD_ATTEMPTED = False


def _warn_once(attr: str, message: str) -> None:
    if getattr(_warn_once, attr, False):
        return
    warnings.warn(message, stacklevel=2)
    setattr(_warn_once, attr, True)


def _load_c_lib() -> Optional[ctypes.CDLL]:
    global _C_LIB, _C_LOAD_ATTEMPTED
    if _C_LOAD_ATTEMPTED:
        return _C_LIB
    _C_LOAD_ATTEMPTED = True
    so_path = Path(__file__).parent / "phase3_core.so"
    if not so_path.exists():
        return None
    try:
        lib = ctypes.CDLL(str(so_path))
        lib.encode_c.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int8),
        ]
        lib.encode_c.restype = ctypes.c_int32
        lib.viterbi_neural_bm_c.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int8),
        ]
        lib.viterbi_neural_bm_c.restype = ctypes.c_int32
        lib.check_connectivity_c.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
        lib.check_connectivity_c.restype = ctypes.c_int32
        lib.check_termination_c.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.check_termination_c.restype = ctypes.c_int32
        lib.check_noncatastrophic_c.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int32,
        ]
        lib.check_noncatastrophic_c.restype = ctypes.c_int32
        lib.compute_dfree_c.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.compute_dfree_c.restype = ctypes.c_int32
        lib.mutate_and_validate_c.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.mutate_and_validate_c.restype = ctypes.c_int32
        _C_LIB = lib
    except OSError:
        _C_LIB = None
    return _C_LIB


def is_available() -> bool:
    return _load_c_lib() is not None


def _encode_py(next_state, output_bits, info_bits, S, K_INFO, memory) -> np.ndarray:
    state = 0
    coded = np.empty(2 * (K_INFO + memory), dtype=np.int8)
    pos = 0
    for t in range(K_INFO):
        bit = int(info_bits[t])
        coded[pos:pos + 2] = output_bits[state, bit]
        pos += 2
        state = int(next_state[state, bit])
    for _ in range(memory):
        coded[pos:pos + 2] = output_bits[state, 0]
        pos += 2
        state = int(next_state[state, 0])
    return coded


def encode_native(next_state, output_bits, info_bits, S, K_INFO, memory) -> np.ndarray:
    lib = _load_c_lib()
    next_state = np.ascontiguousarray(next_state, dtype=np.int32)
    output_bits = np.ascontiguousarray(output_bits, dtype=np.int8)
    info_bits = np.ascontiguousarray(info_bits, dtype=np.int8)
    if lib is None:
        _warn_once("_warn_encode", "phase3_core.so not found; encode_native is using Python fallback.")
        return _encode_py(next_state, output_bits, info_bits, S, K_INFO, memory)
    coded = np.zeros(2 * (K_INFO + memory), dtype=np.int8)
    lib.encode_c(
        next_state.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        output_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        info_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        ctypes.c_int32(S),
        ctypes.c_int32(K_INFO),
        ctypes.c_int32(memory),
        coded.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    return coded


def viterbi_neural_bm_native(branch_metrics, next_state, index_table, S, T, K_INFO) -> np.ndarray:
    lib = _load_c_lib()
    branch_metrics = np.ascontiguousarray(branch_metrics, dtype=np.float64)
    next_state = np.ascontiguousarray(next_state, dtype=np.int32)
    index_table = np.ascontiguousarray(index_table, dtype=np.int32)
    if lib is None:
        _warn_once(
            "_warn_vn",
            "phase3_core.so not found; viterbi_neural_bm_native is using Python fallback.",
        )
        dummy = Trellis(
            n_states=int(S),
            next_state=next_state.reshape(S, 2),
            output_bits=np.zeros((S, 2, 2), dtype=np.int8),
        )
        return viterbi_neural_bm(branch_metrics.reshape(T, 4), dummy, index_table.reshape(S, 2))
    decoded = np.zeros(K_INFO, dtype=np.int8)
    lib.viterbi_neural_bm_c(
        branch_metrics.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        next_state.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        index_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(S),
        ctypes.c_int32(T),
        ctypes.c_int32(K_INFO),
        decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    return decoded


def check_connectivity_native(next_state, S) -> bool:
    lib = _load_c_lib()
    next_state = np.ascontiguousarray(next_state, dtype=np.int32)
    if lib is None:
        _warn_once(
            "_warn_conn",
            "phase3_core.so not found; connectivity checks are using Python fallback.",
        )
        trellis = Trellis(n_states=int(S), next_state=next_state.reshape(S, 2), output_bits=np.zeros((S, 2, 2), dtype=np.int8))
        return is_fully_connected(trellis)
    return bool(
        lib.check_connectivity_c(
            next_state.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(S),
        )
    )


def check_termination_native(next_state, S, max_tail=6) -> bool:
    lib = _load_c_lib()
    next_state = np.ascontiguousarray(next_state, dtype=np.int32)
    if lib is None:
        _warn_once(
            "_warn_term",
            "phase3_core.so not found; termination checks are using Python fallback.",
        )
        trellis = Trellis(n_states=int(S), next_state=next_state.reshape(S, 2), output_bits=np.zeros((S, 2, 2), dtype=np.int8))
        return is_terminating(trellis, max_tail=max_tail)
    return bool(
        lib.check_termination_c(
            next_state.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(S),
            ctypes.c_int32(max_tail),
        )
    )


def check_noncatastrophic_native(next_state, output_bits, S) -> bool:
    lib = _load_c_lib()
    next_state = np.ascontiguousarray(next_state, dtype=np.int32)
    output_bits = np.ascontiguousarray(output_bits, dtype=np.int8)
    if lib is None:
        _warn_once(
            "_warn_nc",
            "phase3_core.so not found; non-catastrophic checks are using Python fallback.",
        )
        trellis = Trellis(n_states=int(S), next_state=next_state.reshape(S, 2), output_bits=output_bits.reshape(S, 2, 2))
        return is_non_catastrophic(trellis)
    return bool(
        lib.check_noncatastrophic_c(
            next_state.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            output_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ctypes.c_int32(S),
        )
    )


def compute_dfree_native(next_state, output_bits, S) -> float:
    lib = _load_c_lib()
    next_state = np.ascontiguousarray(next_state, dtype=np.int32)
    output_bits = np.ascontiguousarray(output_bits, dtype=np.int8)
    if lib is None:
        _warn_once("_warn_df", "phase3_core.so not found; d_free is using Python fallback.")
        trellis = Trellis(n_states=int(S), next_state=next_state.reshape(S, 2), output_bits=output_bits.reshape(S, 2, 2))
        return compute_dfree(trellis)
    dfree_out = ctypes.c_int32()
    rc = lib.compute_dfree_c(
        next_state.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        output_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        ctypes.c_int32(S),
        ctypes.byref(dfree_out),
    )
    return float("inf") if rc != 0 else float(dfree_out.value)


def mutate_and_validate_native(
    parent_genome: TrellisGenome,
    n_edges: int,
    max_attempts: int,
    dfree_target: int,
    seed: int,
) -> Optional[tuple[TrellisGenome, float, int]]:
    lib = _load_c_lib()
    if lib is None:
        _warn_once(
            "_warn_mut",
            "phase3_core.so not found; mutate_and_validate_native is using Python fallback.",
        )
        rng = np.random.default_rng(seed)
        return mutate_and_validate(parent_genome, n_edges, max_attempts, dfree_target, rng)

    next_state = np.ascontiguousarray(parent_genome["next_state"], dtype=np.int32)
    output_pair = np.ascontiguousarray(parent_genome["output_pair"], dtype=np.int32)
    S = int(parent_genome["n_states"])
    out_next_state = np.empty_like(next_state)
    out_output_pair = np.empty_like(output_pair)
    out_output_bits = np.empty((S, 2, 2), dtype=np.int8)
    out_dfree = ctypes.c_int32()
    out_attempts = ctypes.c_int32()
    rc = lib.mutate_and_validate_c(
        next_state.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        output_pair.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(S),
        ctypes.c_int32(n_edges),
        ctypes.c_int32(max_attempts),
        ctypes.c_int32(dfree_target),
        ctypes.c_uint64(seed),
        out_next_state.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_output_pair.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_output_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        ctypes.byref(out_dfree),
        ctypes.byref(out_attempts),
    )
    if rc != 0:
        return None
    child: TrellisGenome = {
        "next_state": out_next_state,
        "output_pair": out_output_pair,
        "n_states": S,
    }
    return child, float(out_dfree.value), int(out_attempts.value)

