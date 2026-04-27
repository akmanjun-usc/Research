from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

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
N_SEEDS = 5
PRINT_EVERY = 10


def _blank_seed_status(seed: int) -> dict:
    return {
        "seed": int(seed),
        "status": "pending",
        "gen": None,
        "best_fitness": None,
        "mean_fitness": None,
        "plateau": None,
        "convergence_generation": None,
    }


def _fmt_int(value: Optional[int]) -> str:
    return "--" if value is None else f"{int(value):d}"


def _fmt_float(value: Optional[float]) -> str:
    return "--" if value is None else f"{float(value):.4e}"


def _render_progress_table(seed_statuses: list[dict]) -> str:
    lines = [
        "seed | status  | gen | best_bler  | mean_bler  | plateau",
        "-----+---------+-----+------------+------------+--------",
    ]
    for status in seed_statuses:
        lines.append(
            f"{status['seed']:>4d} | "
            f"{status['status']:<7} | "
            f"{_fmt_int(status['gen']):>3} | "
            f"{_fmt_float(status['best_fitness']):>10} | "
            f"{_fmt_float(status['mean_fitness']):>10} | "
            f"{_fmt_int(status['plateau']):>6}"
        )
    return "\n".join(lines)


def _should_print_progress(snapshot: dict, every: int = PRINT_EVERY) -> bool:
    if snapshot.get("is_complete", False):
        return True
    generation = int(snapshot["generation"])
    return generation == 0 or (generation + 1) % every == 0


def _update_seed_status(seed_statuses: list[dict], snapshot: dict) -> None:
    idx = int(snapshot["seed"])
    status = seed_statuses[idx]
    status["status"] = "done" if snapshot.get("is_complete", False) else "running"
    status["gen"] = int(snapshot["generation"])
    status["best_fitness"] = float(snapshot["best_fitness"])
    status["mean_fitness"] = float(snapshot["mean_fitness"])
    status["plateau"] = int(snapshot["plateau"])
    status["convergence_generation"] = snapshot.get("convergence_generation")


def main() -> None:
    seed_statuses = [_blank_seed_status(seed) for seed in range(N_SEEDS)]
    print(_render_progress_table(seed_statuses), flush=True)

    for rng_seed in range(N_SEEDS):
        rng = np.random.default_rng(rng_seed)
        init_pop = build_initial_population(
            pop_size=POP_SIZE,
            base_genome=nasa_k7_genome(),
            dfree_target=DFREE_TARGET,
            rng=rng,
        )
        fitness_fn = functools.partial(fitness_oracle, n_trials=N_TRIALS, snr_db=5.0, inr_db=5.0)

        def progress_callback(snapshot: dict) -> None:
            _update_seed_status(seed_statuses, snapshot)
            if _should_print_progress(snapshot, every=PRINT_EVERY):
                print(_render_progress_table(seed_statuses), flush=True)

        result = run_ea(
            fitness_fn=fitness_fn,
            init_population=init_pop,
            n_generations=N_GENERATIONS,
            pop_size=POP_SIZE,
            elite_size=2,
            dfree_target=DFREE_TARGET,
            rng_seed=rng_seed,
            log_path=RESULTS_DIR / f"log_seed{rng_seed}.npz",
            progress_callback=progress_callback,
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
            f"converged at gen {result.get('convergence_generation', N_GENERATIONS)}",
            flush=True,
        )

    print(_render_progress_table(seed_statuses), flush=True)


if __name__ == "__main__":
    main()
