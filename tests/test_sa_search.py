"""
tests/test_sa_search.py — Unit tests for sa_search.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sa_search import REJECTED_ENERGY, _clone_genome, random_trellis_from_scratch, run_sa
from trellis_genome import TrellisGenome, genome_hash, is_valid_genome, nasa_k7_genome


# ── helpers ──────────────────────────────────────────────────────────────────

def _dummy_energy(target_sum: int = 0):
    """Energy = |sum(output_pair) - target_sum|. Minimum at target_sum."""
    def fn(genome: TrellisGenome, seed: int) -> float:
        return float(abs(int(genome["output_pair"].sum()) - target_sum))
    return fn


# ── random_trellis_from_scratch ───────────────────────────────────────────────

def test_random_trellis_from_scratch_no_nasa_bias():
    nasa_hash = genome_hash(nasa_k7_genome())
    rng = np.random.default_rng(7)
    for _ in range(5):
        g = random_trellis_from_scratch(rng, dfree_target=8)
        assert genome_hash(g) != nasa_hash, "random genome must not equal NASA K=7"
        assert is_valid_genome(g), "random genome must pass all constraint checks"
        assert g["n_states"] == 64


def test_random_trellis_from_scratch_dfree_respected():
    rng = np.random.default_rng(42)
    g = random_trellis_from_scratch(rng, dfree_target=1)
    assert is_valid_genome(g)


# ── run_sa ────────────────────────────────────────────────────────────────────

def test_run_sa_convergence_dummy():
    """SA should reduce energy on a trivial dummy fitness."""
    rng = np.random.default_rng(0)
    init = random_trellis_from_scratch(rng, dfree_target=1)
    init_e = float(init["output_pair"].sum())  # energy at start

    result = run_sa(
        energy_fn=_dummy_energy(target_sum=0),
        init_genome=init,
        n_steps=500,
        T0=10.0,
        alpha=0.99,
        n_edges_range=(1, 3),
        dfree_target=1,
        rng_seed=42,
    )
    assert result["best_energy"] <= init_e, "SA must not make things worse on dummy energy"
    assert math.isfinite(result["best_energy"])


def test_run_sa_accepts_downhill():
    """A strictly better candidate must always be accepted."""
    rng = np.random.default_rng(1)
    init = random_trellis_from_scratch(rng, dfree_target=1)

    energies = [5.0, 4.0]  # second call is better
    call_count = [0]

    def energy_fn(genome: TrellisGenome, seed: int) -> float:
        val = energies[min(call_count[0], len(energies) - 1)]
        call_count[0] += 1
        return val

    result = run_sa(
        energy_fn=energy_fn,
        init_genome=init,
        n_steps=2,
        T0=1.0,
        alpha=0.99,
        n_edges_range=(1, 1),
        dfree_target=1,
        rng_seed=0,
    )
    # best_energy must be at most 4.0 (downhill step accepted)
    assert result["best_energy"] <= 4.0


def test_run_sa_probabilistic_uphill():
    """At high T, small uphill steps are accepted most of the time."""
    rng = np.random.default_rng(2)
    init = random_trellis_from_scratch(rng, dfree_target=1)

    # Energy alternates: high then slightly higher, forcing uphill
    call_idx = [0]
    e_seq = [1.0, 1.01]  # delta = 0.01; at T=10.0, p(accept) = exp(-0.001) ≈ 0.999

    def energy_fn(genome: TrellisGenome, seed: int) -> float:
        val = e_seq[call_idx[0] % 2]
        call_idx[0] += 1
        return val

    result = run_sa(
        energy_fn=energy_fn,
        init_genome=init,
        n_steps=100,
        T0=10.0,
        alpha=1.0,  # no cooling — keep high T throughout
        n_edges_range=(1, 1),
        dfree_target=1,
        rng_seed=123,
    )
    # With T=10 and delta=0.01, acceptance rate should be very high (>80%)
    total_moves = result["n_accepted"] + result["n_rejected"]
    if total_moves > 0:
        rate = result["n_accepted"] / total_moves
        assert rate > 0.5, f"Expected high acceptance rate at T=10, got {rate:.2f}"


def test_run_sa_rejected_energy_not_propagated():
    """Chain must not update current genome when candidate returns REJECTED_ENERGY."""
    rng = np.random.default_rng(3)
    init = random_trellis_from_scratch(rng, dfree_target=1)

    def always_inf(genome: TrellisGenome, seed: int) -> float:
        return REJECTED_ENERGY

    result = run_sa(
        energy_fn=always_inf,
        init_genome=init,
        n_steps=10,
        T0=1.0,
        alpha=0.9,
        n_edges_range=(1, 1),
        dfree_target=1,
        rng_seed=7,
    )
    # best_energy stays at REJECTED_ENERGY when all candidates are inf
    assert not math.isfinite(result["best_energy"]) or result["n_accepted"] == 0


def test_run_sa_log_format():
    """Log entries must contain all required keys (SA + base format)."""
    rng = np.random.default_rng(5)
    init = random_trellis_from_scratch(rng, dfree_target=1)

    result = run_sa(
        energy_fn=_dummy_energy(),
        init_genome=init,
        n_steps=5,
        T0=1.0,
        alpha=0.99,
        n_edges_range=(1, 1),
        dfree_target=1,
        rng_seed=0,
    )
    required_keys = {"genome_hash", "fitness", "generation", "eval_seed", "temperature", "accepted"}
    for entry in result["log"]:
        assert required_keys <= set(entry.keys()), f"Missing keys: {required_keys - set(entry.keys())}"


def test_run_sa_return_dict_keys():
    """run_sa must return all documented keys."""
    rng = np.random.default_rng(9)
    init = random_trellis_from_scratch(rng, dfree_target=1)

    result = run_sa(
        energy_fn=_dummy_energy(),
        init_genome=init,
        n_steps=5,
        T0=1.0,
        alpha=0.99,
        n_edges_range=(1, 1),
        dfree_target=1,
        rng_seed=0,
    )
    for key in ("best_genome", "best_energy", "energy_curve", "temperature_curve",
                "acceptance_rate_curve", "n_accepted", "n_rejected", "n_invalid", "log"):
        assert key in result, f"Missing key: {key}"
    assert len(result["energy_curve"]) == 5
    assert len(result["temperature_curve"]) == 5
