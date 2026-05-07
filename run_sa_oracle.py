"""
run_sa_oracle.py — Phase 3c SA search with oracle Viterbi fitness
"""
from __future__ import annotations

import functools
import math
import time
from pathlib import Path

import numpy as np

from sa_fitness import cascaded_fitness_oracle
from sa_search import run_sa_two_phase

RESULTS_DIR = Path("results/phase3c")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Phase 1: aggressive exploration ──────────────────────────────────────────
N_CHAINS = 5         # independent chains from random starts; set to 1 for single-chain
N_STEPS_PHASE1 = 2_000
T0_PHASE1 = 0.05     # high: accepts ~94% uphill at delta=0.003
ALPHA_PHASE1 = 0.998  # T_final ≈ 0.0018 after 2000 steps
N_EDGES_PHASE1 = (3, 5)

# ── Phase 2: refinement ───────────────────────────────────────────────────────
N_STEPS_PHASE2 = 1_000
T0_PHASE2 = 0.005
ALPHA_PHASE2 = 0.997
N_EDGES_PHASE2 = (1, 2)

DFREE_TARGET = 8
N_SEEDS = 1

# ── Logging frequency ─────────────────────────────────────────────────────────
LOG_EVERY = 100  # print progress every N steps


_HDR = (
    f"{'step':>6} | {'cur_energy':>11} | {'best_energy':>11} | "
    f"{'temp':>8} | {'acc':>5} | {'rej':>5} | {'inv':>5} | {'elapsed':>8}"
)
_SEP = "-" * len(_HDR)


def _fmt_e(v: float) -> str:
    return f"{v:.4e}" if math.isfinite(v) else "    inf   "


def _make_step_callback(phase_label: str) -> callable:
    last_print = [-1]

    def callback(snap: dict) -> None:
        step = snap["step"]
        if step % LOG_EVERY == 0 or step == 0:
            if last_print[0] < 0:
                print(f"\n{phase_label}", flush=True)
                print(_HDR, flush=True)
                print(_SEP, flush=True)
            last_print[0] = step
            print(
                f"{step:>6d} | "
                f"{_fmt_e(snap['current_energy']):>11} | "
                f"{_fmt_e(snap['best_energy']):>11} | "
                f"{snap['temperature']:>8.5f} | "
                f"{snap['n_accepted']:>5d} | "
                f"{snap['n_rejected']:>5d} | "
                f"{snap['n_invalid']:>5d} | "
                f"{snap['elapsed']:>7.1f}s",
                flush=True,
            )

    return callback


def main() -> None:
    energy_fn = functools.partial(cascaded_fitness_oracle)

    for rng_seed in range(N_SEEDS):
        print(f"\n{'='*70}", flush=True)
        print(
            f"run_sa_oracle  seed={rng_seed}  "
            f"chains={N_CHAINS}  steps_p1={N_STEPS_PHASE1}  steps_p2={N_STEPS_PHASE2}  "
            f"SNR=10/6/3 dB  INR=5 dB",
            flush=True,
        )
        print(f"{'='*70}", flush=True)

        t_start = time.perf_counter()

        result = None
        try:
            result = run_sa_two_phase(
                energy_fn_phase1=energy_fn,
                energy_fn_phase2=energy_fn,
                n_chains=N_CHAINS,
                n_steps_phase1=N_STEPS_PHASE1,
                n_steps_phase2=N_STEPS_PHASE2,
                T0_phase1=T0_PHASE1,
                alpha_phase1=ALPHA_PHASE1,
                T0_phase2=T0_PHASE2,
                alpha_phase2=ALPHA_PHASE2,
                n_edges_phase1=N_EDGES_PHASE1,
                n_edges_phase2=N_EDGES_PHASE2,
                dfree_target=DFREE_TARGET,
                rng_seed=rng_seed,
                log_dir=RESULTS_DIR,
                step_callback_phase1=_make_step_callback(
                    f"[Phase 1 progress — seed {rng_seed}]"
                ),
                step_callback_phase2=_make_step_callback(
                    f"[Phase 2 progress — seed {rng_seed}]"
                ),
            )
        finally:
            if result is not None:
                np.savez(
                    RESULTS_DIR / f"best_trellis_seed{rng_seed}.npz",
                    next_state=result["best_genome"]["next_state"],
                    output_pair=result["best_genome"]["output_pair"],
                    fitness=result["best_energy"],
                    seed=rng_seed,
                )
                np.savez(
                    RESULTS_DIR / f"fitness_curves_seed{rng_seed}.npz",
                    best_per_gen=result["energy_curve"],
                    temperature_per_step=result["temperature_curve"],
                    acceptance_rate=result["acceptance_rate_curve"],
                    phase1_best_energy=result["phase1_best_energy"],
                )
                elapsed = time.perf_counter() - t_start
                be = result["best_energy"]
                print(
                    f"\nSeed {rng_seed} complete in {elapsed:.0f}s  |  "
                    f"phase1_best={result['phase1_best_energy']:.4e}  "
                    f"final_best={f'{be:.4e}' if math.isfinite(be) else 'inf'}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
