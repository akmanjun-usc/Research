"""
search.py — Evolutionary search engine for trellis genomes
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from constraints import compute_dfree
from phase3_native import mutate_and_validate_native
from trellis_genome import (
    TrellisGenome,
    genome_hash,
    genome_to_trellis,
    is_valid_genome,
    mutate_and_validate,
)


def _clone_genome(genome: TrellisGenome) -> TrellisGenome:
    return {
        "next_state": np.array(genome["next_state"], dtype=np.int32, copy=True),
        "output_pair": np.array(genome["output_pair"], dtype=np.int32, copy=True),
        "n_states": int(genome["n_states"]),
    }


def build_initial_population(
    pop_size: int,
    base_genome: TrellisGenome,
    dfree_target: int,
    rng: np.random.Generator,
) -> list[TrellisGenome]:
    population = [_clone_genome(base_genome)]
    cut1 = pop_size // 3
    cut2 = 2 * (pop_size // 3)
    for idx in range(1, pop_size):
        if idx < cut1:
            n_edges = 1
        elif idx < cut2:
            n_edges = 2
        else:
            n_edges = 3
        result = mutate_and_validate(
            base_genome,
            n_edges=n_edges,
            max_attempts=10_000,
            dfree_target=dfree_target,
            rng=rng,
        )
        if result is None:
            raise RuntimeError(f"failed to build initial population member {idx}")
        population.append(result[0])
    return population


def _tournament(population, fitnesses, tournament_size, rng) -> TrellisGenome:
    indices = rng.integers(0, len(population), size=tournament_size)
    best = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if fitnesses[idx] < fitnesses[best]:
            best = idx
    return population[best]


def _statewise_crossover(parent_a: TrellisGenome, parent_b: TrellisGenome, rng: np.random.Generator) -> TrellisGenome:
    child = _clone_genome(parent_a)
    S = int(child["n_states"])
    mask = rng.integers(0, 2, size=S, dtype=np.int8).astype(bool)
    if not np.any(mask):
        mask[int(rng.integers(0, S))] = True
    child["next_state"][mask] = parent_b["next_state"][mask]
    child["output_pair"][mask] = parent_b["output_pair"][mask]
    if not is_valid_genome(child):
        return _clone_genome(parent_a if bool(rng.integers(0, 2)) else parent_b)
    return child


def _save_log(log_entries: list[dict], path: Path) -> None:
    if not log_entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        genome_hash=np.array([entry["genome_hash"] for entry in log_entries], dtype=object),
        fitness=np.array([entry["fitness"] for entry in log_entries], dtype=np.float64),
        generation=np.array([entry["generation"] for entry in log_entries], dtype=np.int32),
        eval_seed=np.array([entry["eval_seed"] for entry in log_entries], dtype=np.int64),
    )


def run_ea(
    fitness_fn: Callable[[TrellisGenome, int], float],
    init_population: list[TrellisGenome],
    n_generations: int = 200,
    pop_size: int = 50,
    elite_size: int = 2,
    n_edges_range: tuple[int, int] = (1, 3),
    dfree_target: int = 8,
    tournament_size: int = 3,
    plateau_patience: int = 50,
    rng_seed: int = 0,
    log_path: Optional[Path] = None,
) -> dict:
    if len(init_population) != pop_size:
        raise ValueError(f"expected init_population of size {pop_size}, got {len(init_population)}")

    rng = np.random.default_rng(rng_seed)
    population = [_clone_genome(g) for g in init_population]
    fitnesses = np.full(pop_size, np.inf, dtype=np.float64)
    best_curve: list[float] = []
    mean_curve: list[float] = []
    std_curve: list[float] = []
    log_entries: list[dict] = []
    best_so_far = np.inf
    best_genome = _clone_genome(population[0])
    convergence_generation: Optional[int] = None
    plateau = 0

    for gen in range(n_generations):
        elite_idx = np.argsort(fitnesses)[:elite_size] if gen > 0 else np.arange(elite_size)
        elite_mask = np.zeros(pop_size, dtype=bool)
        elite_mask[np.asarray(elite_idx, dtype=np.int32)] = True

        for idx, genome in enumerate(population):
            if gen > 0 and elite_mask[idx] and np.isfinite(fitnesses[idx]):
                continue
            eval_seed = int(rng.integers(0, 2**31 - 1))
            fitness = float(fitness_fn(genome, eval_seed))
            fitnesses[idx] = fitness
            log_entries.append(
                {
                    "genome_hash": genome_hash(genome),
                    "fitness": fitness,
                    "generation": gen,
                    "eval_seed": eval_seed,
                }
            )

        gen_best_idx = int(np.argmin(fitnesses))
        gen_best = float(fitnesses[gen_best_idx])
        gen_mean = float(np.mean(fitnesses))
        gen_std = float(np.std(fitnesses))
        best_curve.append(gen_best)
        mean_curve.append(gen_mean)
        std_curve.append(gen_std)

        if gen_best < best_so_far:
            best_so_far = gen_best
            best_genome = _clone_genome(population[gen_best_idx])
            convergence_generation = gen
            plateau = 0
        else:
            plateau += 1

        if log_path is not None and (gen + 1) % 10 == 0:
            _save_log(log_entries, Path(log_path))

        if best_so_far <= 0.0:
            break

        if plateau >= plateau_patience:
            break

        ranked = np.argsort(fitnesses)
        next_population = [_clone_genome(population[int(i)]) for i in ranked[:elite_size]]
        next_fitnesses = np.array([fitnesses[int(i)] for i in ranked[:elite_size]], dtype=np.float64)

        while len(next_population) < pop_size:
            parent_a = _tournament(population, fitnesses, tournament_size, rng)
            parent_b = _tournament(population, fitnesses, tournament_size, rng)
            if rng.random() < 0.3:
                child_seed = parent_a if rng.random() < 0.5 else parent_b
                child = _clone_genome(child_seed)
            else:
                child = _statewise_crossover(parent_a, parent_b, rng)

            n_edges = int(rng.integers(n_edges_range[0], n_edges_range[1] + 1))
            seed = int(rng.integers(0, 2**63 - 1))
            mutated = mutate_and_validate_native(
                child,
                n_edges=n_edges,
                max_attempts=1000,
                dfree_target=dfree_target,
                seed=seed,
            )
            if mutated is None:
                continue
            next_population.append(mutated[0])
            next_fitnesses = np.append(next_fitnesses, np.inf)

        population = next_population
        fitnesses = next_fitnesses

    if log_path is not None:
        _save_log(log_entries, Path(log_path))

    return {
        "best_genome": best_genome,
        "best_fitness": best_so_far,
        "fitness_curves": {
            "best": np.array(best_curve, dtype=np.float64),
            "mean": np.array(mean_curve, dtype=np.float64),
            "std": np.array(std_curve, dtype=np.float64),
        },
        "population": population,
        "log": log_entries,
        "convergence_generation": convergence_generation if convergence_generation is not None else n_generations,
    }
