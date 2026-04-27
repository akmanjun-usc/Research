from __future__ import annotations

import functools
import time
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
N_SEEDS = 1




_GEN_HEADER = (
    f"{'gen':>4} | {'best_bler':>10} | {'mean_bler':>10} | {'std_bler':>9} | "
    f"{'plateau':>7} | {'n_eval':>6} | {'eval_s':>7} | {'off_s':>6} | {'tot_s':>6}"
)
_GEN_SEP = "-" * len(_GEN_HEADER)


def _gen_line(snap: dict) -> str:
    return (
        f"{snap['generation']:>4d} | "
        f"{snap['best_fitness']:>10.4e} | "
        f"{snap['mean_fitness']:>10.4e} | "
        f"{snap['std_fitness']:>9.4e} | "
        f"{snap['plateau']:>7d} | "
        f"{snap['n_evaluated']:>6d} | "
        f"{snap['evaluation_time_s']:>7.1f} | "
        f"{snap['offspring_time_s']:>6.1f} | "
        f"{snap['generation_time_s']:>6.1f}"
    )


def main() -> None:
    for rng_seed in range(N_SEEDS):
        print(f"\n{'='*60}", flush=True)
        print(f"Seed {rng_seed}  |  pop={POP_SIZE}  gen={N_GENERATIONS}  trials={N_TRIALS}  SNR=3dB  INR=5dB", flush=True)
        print(f"{'='*60}", flush=True)

        rng = np.random.default_rng(rng_seed)
        print("Building initial population...", flush=True)
        t0 = time.perf_counter()
        init_pop = build_initial_population(
            pop_size=POP_SIZE,
            base_genome=nasa_k7_genome(),
            dfree_target=DFREE_TARGET,
            rng=rng,
        )
        print(f"  done in {time.perf_counter() - t0:.1f}s\n", flush=True)

        fitness_fn = functools.partial(fitness_oracle, n_trials=N_TRIALS, snr_db=3.0, inr_db=5.0)

        print(_GEN_HEADER, flush=True)
        print(_GEN_SEP, flush=True)

        def generation_callback(snap: dict) -> None:
            print(_gen_line(snap), flush=True)
            if snap.get("is_complete"):
                print(_GEN_SEP, flush=True)
                print(f"  stopped: plateau={snap['plateau']} >= {50}", flush=True)

        result = run_ea(
            fitness_fn=fitness_fn,
            init_population=init_pop,
            n_generations=N_GENERATIONS,
            pop_size=POP_SIZE,
            elite_size=2,
            dfree_target=DFREE_TARGET,
            rng_seed=rng_seed,
            log_path=RESULTS_DIR / f"log_seed{rng_seed}.npz",
            generation_callback=generation_callback,
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
            f"\nSeed {rng_seed} done: best BLER={result['best_fitness']:.4e}, "
            f"converged at gen {result.get('convergence_generation', 'N/A')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
