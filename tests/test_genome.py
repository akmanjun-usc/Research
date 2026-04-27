import numpy as np

from trellis import load_nasa_k7
from trellis_genome import (
    deserialize,
    genome_hash,
    genome_to_trellis,
    is_valid_genome,
    mutate_and_validate,
    nasa_k7_genome,
    perturb,
    random_valid_genome,
    serialize,
    trellis_to_genome,
)


def test_nasa_k7_genome_converts_correctly():
    trellis = load_nasa_k7()
    genome = trellis_to_genome(trellis)
    restored = genome_to_trellis(genome)
    assert np.array_equal(restored.next_state, trellis.next_state)
    assert np.array_equal(restored.output_bits, trellis.output_bits)


def test_serialize_deserialize_roundtrip():
    genome = nasa_k7_genome()
    restored = deserialize(serialize(genome))
    assert restored["n_states"] == genome["n_states"]
    assert np.array_equal(restored["next_state"], genome["next_state"])
    assert np.array_equal(restored["output_pair"], genome["output_pair"])


def test_genome_hash_stable():
    genome = nasa_k7_genome()
    assert genome_hash(genome) == genome_hash(genome)


def test_perturb_produces_different_genome():
    rng = np.random.default_rng(0)
    genome = nasa_k7_genome()
    child = perturb(genome, n_edges=4, rng=rng)
    changed = np.any(child["next_state"] != genome["next_state"]) or np.any(
        child["output_pair"] != genome["output_pair"]
    )
    assert changed


def test_perturb_validity_rate():
    rng = np.random.default_rng(1)
    genome = nasa_k7_genome()
    valid = 0
    trials = 100
    for _ in range(trials):
        if is_valid_genome(perturb(genome, n_edges=1, rng=rng)):
            valid += 1
    assert valid >= 50


def test_random_valid_genome_passes_all():
    rng = np.random.default_rng(2)
    for _ in range(3):
        genome = random_valid_genome(rng=rng, max_tries=100_000)
        assert genome is not None
        assert is_valid_genome(genome)


def test_mutate_and_validate_succeeds():
    rng = np.random.default_rng(3)
    result = mutate_and_validate(
        parent=nasa_k7_genome(),
        n_edges=1,
        max_attempts=1000,
        dfree_target=8,
        rng=rng,
    )
    assert result is not None
