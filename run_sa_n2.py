"""
run_sa_n2.py — Phase 3c SA search with N2 neural branch metric fitness

Phase 1: cascaded BLER filtering via N2 decoder (fast rejection of bad codes)
Phase 2: pure BM alignment energy (combined_energy) — structural proxy for BLER
"""
from __future__ import annotations

import functools
import math
import time
from pathlib import Path

import numpy as np
import torch

from neural_bm import load_model
from sa_fitness import cascaded_fitness_n2, cascaded_fitness_oracle, combined_energy
from sa_search import run_sa_two_phase

RESULTS_DIR = Path("results/phase3c_n2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = Path("results/phase2b/checkpoints/best_model_seed32.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Phase 1: aggressive exploration (N2 cascaded BLER) ───────────────────────
N_CHAINS = 5
N_STEPS_PHASE1 = 2_000
T0_PHASE1 = 0.5       # accepts ~82% uphill at delta=0.1, ~67% at delta=0.2
ALPHA_PHASE1 = 0.9985  # T_final ≈ 0.050 after 2000 steps
N_EDGES_PHASE1 = (3, 5)

# ── Phase 2: BM alignment refinement ─────────────────────────────────────────
N_STEPS_PHASE2 = 1_000
T0_PHASE2 = 0.005
ALPHA_PHASE2 = 0.997
N_EDGES_PHASE2 = (1, 2)
N_ALIGN_PHASE2 = 100   # signals used per combined_energy evaluation

DFREE_TARGET = 8
N_SEEDS = 1

LOG_EVERY = 100


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
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"N2 checkpoint not found: {CHECKPOINT}\n"
            "Run: python neural_bm.py --train --seed 32"
        )

    print(f"Loading N2 model from {CHECKPOINT} (device={DEVICE})...", flush=True)
    model, ckpt = load_model(CHECKPOINT)
    model = model.to(DEVICE)
    model.eval()
    print(
        f"  N2 loaded (seed={ckpt.get('seed', '?')}, "
        f"val_bler={ckpt.get('val_bler', float('nan')):.4f})\n",
        flush=True,
    )

    energy_fn_phase1 = functools.partial(
        cascaded_fitness_oracle,
        stage3_snr_db=5.0,  # 5dB: oracle separates codes more cleanly than 3dB
    )
    energy_fn_phase2 = functools.partial(
        combined_energy,
        model=model,
        device=DEVICE,
        n_align=N_ALIGN_PHASE2,
        snr_db=5.0,
        inr_db=5.0,
    )

    for rng_seed in range(N_SEEDS):
        print(f"\n{'='*70}", flush=True)
        print(
            f"run_sa_n2  seed={rng_seed}  chains={N_CHAINS}  "
            f"steps_p1={N_STEPS_PHASE1}  steps_p2={N_STEPS_PHASE2}  "
            f"p1=oracle-cascade  p2=BM-alignment  device={DEVICE}",
            flush=True,
        )
        print(f"{'='*70}", flush=True)

        t_start = time.perf_counter()

        result = None
        try:
            result = run_sa_two_phase(
                energy_fn_phase1=energy_fn_phase1,
                energy_fn_phase2=energy_fn_phase2,
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
                    f"[Phase 1 (oracle-cascade) — seed {rng_seed}]"
                ),
                step_callback_phase2=_make_step_callback(
                    f"[Phase 2 (BM-alignment) — seed {rng_seed}]"
                ),
                adaptive_restart_phase1=False,  # let temperature cool naturally in Phase 1
                adaptive_restart_phase2=True,
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
