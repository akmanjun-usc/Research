"""
trellis_genome.py — Genome representation for Phase 3 trellis search
"""

from __future__ import annotations

import hashlib
from typing import Optional, TypedDict

import numpy as np

from constraints import compute_dfree, is_fully_connected, is_non_catastrophic, is_terminating
from trellis import Trellis, load_nasa_k7


class TrellisGenome(TypedDict):
    next_state: np.ndarray
    output_pair: np.ndarray
    n_states: int


BITS_FROM_PAIR = np.array(
    [
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
    ],
    dtype=np.int8,
)

_PAIR_FROM_BITS = {
    (1, 1): 0,
    (1, 0): 1,
    (0, 1): 2,
    (0, 0): 3,
}
_SERIALIZE_VERSION = b"TG01"


def genome_to_trellis(genome: TrellisGenome) -> Trellis:
    next_state = np.ascontiguousarray(genome["next_state"], dtype=np.int32)
    output_pair = np.ascontiguousarray(genome["output_pair"], dtype=np.int32)
    output_bits = BITS_FROM_PAIR[output_pair]
    return Trellis(
        n_states=int(genome["n_states"]),
        next_state=next_state,
        output_bits=np.ascontiguousarray(output_bits, dtype=np.int8),
        name="genome_trellis",
    )


def trellis_to_genome(trellis: Trellis) -> TrellisGenome:
    output_pair = np.zeros((trellis.n_states, 2), dtype=np.int32)
    for s in range(trellis.n_states):
        for b in range(2):
            pair = tuple(int(x) for x in trellis.output_bits[s, b])
            output_pair[s, b] = _PAIR_FROM_BITS[pair]
    return {
        "next_state": np.ascontiguousarray(trellis.next_state, dtype=np.int32).copy(),
        "output_pair": output_pair,
        "n_states": int(trellis.n_states),
    }


def nasa_k7_genome() -> TrellisGenome:
    return trellis_to_genome(load_nasa_k7())


def serialize(genome: TrellisGenome) -> bytes:
    next_state = np.ascontiguousarray(genome["next_state"], dtype=np.int32)
    output_pair = np.ascontiguousarray(genome["output_pair"], dtype=np.int32)
    return _SERIALIZE_VERSION + next_state.tobytes() + output_pair.tobytes()


def deserialize(b: bytes) -> TrellisGenome:
    if len(b) < 4 or b[:4] != _SERIALIZE_VERSION:
        raise ValueError("invalid genome serialization header")
    payload = memoryview(b)[4:]
    s = 64
    next_bytes = s * 2 * np.dtype(np.int32).itemsize
    pair_bytes = s * 2 * np.dtype(np.int32).itemsize
    expected = next_bytes + pair_bytes
    if len(payload) != expected:
        raise ValueError(f"invalid genome serialization length: {len(payload)}")
    next_state = np.frombuffer(payload[:next_bytes], dtype=np.int32).copy().reshape(s, 2)
    output_pair = np.frombuffer(payload[next_bytes:], dtype=np.int32).copy().reshape(s, 2)
    return {"next_state": next_state, "output_pair": output_pair, "n_states": s}


def genome_hash(genome: TrellisGenome) -> str:
    return hashlib.sha256(serialize(genome)).hexdigest()


def is_valid_genome(genome: TrellisGenome) -> bool:
    trellis = genome_to_trellis(genome)
    if not is_fully_connected(trellis):
        return False
    if not is_terminating(trellis):
        return False
    if not is_non_catastrophic(trellis):
        return False
    return np.isfinite(compute_dfree(trellis))


def perturb(genome: TrellisGenome, n_edges: int, rng: np.random.Generator) -> TrellisGenome:
    child = {
        "next_state": np.array(genome["next_state"], dtype=np.int32, copy=True),
        "output_pair": np.array(genome["output_pair"], dtype=np.int32, copy=True),
        "n_states": int(genome["n_states"]),
    }
    S = child["n_states"]
    for _ in range(n_edges):
        s = int(rng.integers(0, S))
        b = int(rng.integers(0, 2))
        if bool(rng.integers(0, 2)):
            child["next_state"][s, b] = int(rng.integers(0, S))
        else:
            child["output_pair"][s, b] = int(rng.integers(0, 4))
    return child


def random_valid_genome(
    rng: np.random.Generator,
    max_tries: int = 100_000,
) -> Optional[TrellisGenome]:
    base = nasa_k7_genome()
    attempts_left = max_tries
    while attempts_left > 0:
        budget = min(1000, attempts_left)
        result = mutate_and_validate(
            base,
            n_edges=int(rng.integers(8, 33)),
            max_attempts=budget,
            dfree_target=1,
            rng=rng,
        )
        if result is not None:
            return result[0]
        attempts_left -= budget
    return None


def mutate_and_validate(
    parent: TrellisGenome,
    n_edges: int,
    max_attempts: int,
    dfree_target: int,
    rng: np.random.Generator,
) -> Optional[tuple[TrellisGenome, float, int]]:
    for attempt in range(1, max_attempts + 1):
        child = perturb(parent, n_edges=n_edges, rng=rng)
        trellis = genome_to_trellis(child)
        if not is_fully_connected(trellis):
            continue
        if not is_terminating(trellis):
            continue
        if not is_non_catastrophic(trellis):
            continue
        dfree = compute_dfree(trellis)
        if np.isfinite(dfree) and dfree >= dfree_target:
            return child, float(dfree), attempt
    return None


if __name__ == "__main__":
    nasa = nasa_k7_genome()
    roundtrip = deserialize(serialize(nasa))
    assert np.array_equal(roundtrip["next_state"], nasa["next_state"])
    assert np.array_equal(roundtrip["output_pair"], nasa["output_pair"])
    assert is_valid_genome(nasa)
    print("serialize/deserialize roundtrip OK")
    print("NASA K=7 genome is valid")
