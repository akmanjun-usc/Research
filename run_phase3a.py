from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from fitness import fitness_oracle
from search import build_initial_population, run_ea
from trellis_genome import nasa_k7_genome

RESULTS_DIR = Path("results/phase3a")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

POP_SIZE = 50
N_GENERATIONS = 200
DFREE_TARGET = 8
N_TRIALS = 1000

for rng_seed in range(5):
    rng = np.random.default_rng(rng_seed)
    init_pop = build_initial_population(
        pop_size=POP_SIZE,
        base_genome=nasa_k7_genome(),
        dfree_target=DFREE_TARGET,
        rng=rng,
    )
    fitness_fn = functools.partial(fitness_oracle, n_trials=N_TRIALS, snr_db=5.0, inr_db=5.0)
    result = run_ea(
        fitness_fn=fitness_fn,
        init_population=init_pop,
        n_generations=N_GENERATIONS,
        pop_size=POP_SIZE,
        elite_size=2,
        dfree_target=DFREE_TARGET,
        rng_seed=rng_seed,
        log_path=RESULTS_DIR / f"log_seed{rng_seed}.npz",
    )
    np.savez(
        RESULTS_DIR / f"best_trellis_seed{rng_seed}.npz",
        next_state=result["best_genome"]["next_state"],
        output_pair=result["best_genome"]["output_pair"],
        fitness=result["best_fitness"],
        seed=rng_seed,
    )
    np.savez(
        RESULTS_DIR / f"fitness_curves_seed{rng_seed}.npz",
        best_per_gen=result["fitness_curves"]["best"],
        mean_per_gen=result["fitness_curves"]["mean"],
        std_per_gen=result["fitness_curves"]["std"],
    )
    print(
        f"Seed {rng_seed}: best BLER={result['best_fitness']:.4e}, "
        f"converged at gen {result.get('convergence_generation', N_GENERATIONS)}"
    )
