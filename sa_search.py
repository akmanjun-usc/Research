"""
sa_search.py — Simulated Annealing engine for trellis code search (Phase 3c)
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

import numpy as np

from phase3_native import (
    check_connectivity_native,
    check_noncatastrophic_native,
    check_termination_native,
    compute_dfree_native,
    mutate_and_validate_native,
)
from trellis_genome import BITS_FROM_PAIR, TrellisGenome, genome_hash

REJECTED_ENERGY: float = float("inf")
_N_STATES = 64  # K_c=7, memory=6


def _clone_genome(genome: TrellisGenome) -> TrellisGenome:
    return {
        "next_state": genome["next_state"].copy(),
        "output_pair": genome["output_pair"].copy(),
        "n_states": int(genome["n_states"]),
    }


def random_trellis_from_scratch(
    rng: np.random.Generator,
    dfree_target: int = 8,
    max_tries: int = 200_000,
) -> TrellisGenome:
    """
    Generate a truly random valid trellis genome — no NASA K=7 ancestry.

    Fix next_state to the standard shift-register structure (guarantees full
    connectivity and termination automatically), then sample output_pair
    uniformly from {0,1,2,3} and check non-catastrophic + dfree >= dfree_target.

    This randomizes the generator polynomials (the code search space) while
    keeping the structural constraints cheap to satisfy. Distinct from
    random_valid_genome() in trellis_genome.py which perturbs nasa_k7_genome().
    """
    S = _N_STATES
    half = S >> 1  # 32

    # Standard shift-register next_state: state s = [b1..b6]
    #   input=0: next = s >> 1          (shift in 0 from MSB)
    #   input=1: next = (s >> 1) | half (shift in 1 from MSB)
    # This guarantees: fully_connected and terminating in <= 6 zero-input steps.
    ns = np.zeros((S, 2), dtype=np.int32)
    ns[:, 0] = np.arange(S, dtype=np.int32) >> 1
    ns[:, 1] = (np.arange(S, dtype=np.int32) >> 1) | half
    ns_c = np.ascontiguousarray(ns, dtype=np.int32)

    for _ in range(max_tries):
        output_pair = rng.integers(0, 4, size=(S, 2), dtype=np.int32)
        output_bits = np.ascontiguousarray(BITS_FROM_PAIR[output_pair], dtype=np.int8)

        if not check_noncatastrophic_native(ns_c, output_bits, S):
            continue
        dfree = compute_dfree_native(ns_c, output_bits, S)
        if math.isfinite(dfree) and dfree >= dfree_target:
            return {"next_state": ns.copy(), "output_pair": output_pair, "n_states": S}

    raise RuntimeError(
        f"random_trellis_from_scratch: failed after {max_tries} tries "
        f"(dfree_target={dfree_target})"
    )


def _save_sa_log(log_entries: list[dict], path: Path) -> None:
    if not log_entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        genome_hash=np.array([e["genome_hash"] for e in log_entries], dtype=object),
        fitness=np.array([e["fitness"] for e in log_entries], dtype=np.float64),
        generation=np.array([e["generation"] for e in log_entries], dtype=np.int32),
        eval_seed=np.array([e["eval_seed"] for e in log_entries], dtype=np.int64),
        temperature=np.array([e["temperature"] for e in log_entries], dtype=np.float64),
        accepted=np.array([e["accepted"] for e in log_entries], dtype=np.int8),
    )


def run_sa(
    energy_fn: Callable[[TrellisGenome, int], float],
    init_genome: TrellisGenome,
    n_steps: int,
    T0: float,
    alpha: float,
    n_edges_range: tuple[int, int],
    dfree_target: int,
    rng_seed: int,
    log_path: Optional[Path] = None,
    step_callback: Optional[Callable[[dict], None]] = None,
    adaptive_restart: bool = True,
    restart_acceptance_threshold: float = 0.01,
    restart_patience: int = 10,
    restart_T_fraction: float = 0.5,
    max_mutation_attempts: int = 1000,
) -> dict:
    """
    Single-chain SA with exponential cooling and adaptive restarts.

    Metropolis criterion:
      - candidate = inf   → always reject
      - current   = inf   → always accept any finite candidate (escape)
      - otherwise: accept if delta <= 0 or rng.random() < exp(-delta / T)

    Adaptive restart: if acceptance rate over last `restart_patience` steps
    is below `restart_acceptance_threshold`, bump T back up by
    T = max(T, T0 * restart_T_fraction) and clear the window.

    Returns dict with keys: best_genome, best_energy, energy_curve,
    temperature_curve, acceptance_rate_curve, n_accepted, n_rejected,
    n_invalid, log.
    """
    rng = np.random.default_rng(rng_seed)

    current_genome = _clone_genome(init_genome)
    init_seed = int(rng.integers(0, 2**31 - 1))
    current_energy = energy_fn(current_genome, init_seed)

    best_genome = _clone_genome(current_genome)
    best_energy = current_energy

    T = T0
    n_accepted = 0
    n_rejected = 0
    n_invalid = 0

    energy_curve = np.full(n_steps, REJECTED_ENERGY, dtype=np.float64)
    temperature_curve = np.zeros(n_steps, dtype=np.float64)
    log_entries: list[dict] = []
    acceptance_window: deque = deque(maxlen=restart_patience)

    t0_wall = perf_counter()

    for step in range(n_steps):
        n_edges = int(rng.integers(n_edges_range[0], n_edges_range[1] + 1))
        mut_seed = int(rng.integers(0, 2**63 - 1))
        result = mutate_and_validate_native(
            current_genome, n_edges, max_mutation_attempts, dfree_target, mut_seed
        )

        if result is None:
            n_invalid += 1
            energy_curve[step] = current_energy
            temperature_curve[step] = T
            T *= alpha
            continue

        candidate_genome, _, _ = result
        eval_seed = int(rng.integers(0, 2**31 - 1))
        candidate_energy = energy_fn(candidate_genome, eval_seed)

        # Metropolis
        if not math.isfinite(candidate_energy):
            accepted = False
        elif not math.isfinite(current_energy):
            accepted = True
        else:
            delta = candidate_energy - current_energy
            accepted = delta <= 0.0 or rng.random() < math.exp(-delta / T)

        if accepted:
            current_genome = candidate_genome
            current_energy = candidate_energy
            n_accepted += 1
        else:
            n_rejected += 1

        acceptance_window.append(int(accepted))

        if math.isfinite(current_energy) and current_energy < best_energy:
            best_energy = current_energy
            best_genome = _clone_genome(current_genome)

        if adaptive_restart and len(acceptance_window) == restart_patience:
            rate = sum(acceptance_window) / restart_patience
            if rate < restart_acceptance_threshold:
                T = max(T, T0 * restart_T_fraction)
                acceptance_window.clear()

        log_entries.append({
            "genome_hash": genome_hash(current_genome),
            "fitness": current_energy,
            "generation": step,
            "eval_seed": eval_seed,
            "temperature": T,
            "accepted": int(accepted),
        })

        if step_callback is not None:
            step_callback({
                "step": step,
                "current_energy": current_energy,
                "best_energy": best_energy,
                "temperature": T,
                "n_accepted": n_accepted,
                "n_rejected": n_rejected,
                "n_invalid": n_invalid,
                "elapsed": perf_counter() - t0_wall,
            })

        energy_curve[step] = current_energy
        temperature_curve[step] = T
        T *= alpha

    # Rolling acceptance rate curve (window=50)
    accepted_arr = np.array([e["accepted"] for e in log_entries], dtype=np.float64)
    window = min(50, len(accepted_arr))
    if len(accepted_arr) >= window and window > 0:
        kernel = np.ones(window) / window
        acceptance_rate_curve = np.convolve(accepted_arr, kernel, mode="valid")
    else:
        acceptance_rate_curve = accepted_arr.copy()

    if log_path is not None:
        _save_sa_log(log_entries, log_path)

    return {
        "best_genome": best_genome,
        "best_energy": best_energy,
        "energy_curve": energy_curve,
        "temperature_curve": temperature_curve,
        "acceptance_rate_curve": acceptance_rate_curve,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "n_invalid": n_invalid,
        "log": log_entries,
    }


def run_sa_two_phase(
    energy_fn_phase1: Callable[[TrellisGenome, int], float],
    energy_fn_phase2: Callable[[TrellisGenome, int], float],
    n_chains: int,
    n_steps_phase1: int,
    n_steps_phase2: int,
    T0_phase1: float,
    alpha_phase1: float,
    T0_phase2: float,
    alpha_phase2: float,
    n_edges_phase1: tuple[int, int],
    n_edges_phase2: tuple[int, int],
    dfree_target: int,
    rng_seed: int,
    log_dir: Optional[Path] = None,
    step_callback_phase1: Optional[Callable[[dict], None]] = None,
    step_callback_phase2: Optional[Callable[[dict], None]] = None,
    adaptive_restart_phase1: bool = True,
    adaptive_restart_phase2: bool = True,
) -> dict:
    """
    Two-phase SA: n_chains independent Phase-1 chains from random starts,
    then one Phase-2 refinement chain from the single best Phase-1 genome.

    Works identically for n_chains=1 (single chain) or any n_chains > 1.
    Phase 1 uses energy_fn_phase1 (BLER-based cascaded fitness).
    Phase 2 uses energy_fn_phase2 (BM alignment or finer fitness).
    """
    rng = np.random.default_rng(rng_seed)

    # ── Phase 1: n_chains independent chains ──────────────────────────────────
    phase1_results = []
    for i in range(n_chains):
        chain_seed = int(rng.integers(0, 2**31 - 1))
        print(f"[Phase 1] Generating random trellis for chain {i+1}/{n_chains}...", flush=True)
        init_genome = random_trellis_from_scratch(rng, dfree_target=dfree_target)

        log_path = None
        if log_dir is not None:
            log_path = log_dir / f"log_phase1_chain{i}_seed{rng_seed}.npz"

        print(
            f"[Phase 1] Chain {i+1}/{n_chains}: "
            f"{n_steps_phase1} steps, T0={T0_phase1}, alpha={alpha_phase1}",
            flush=True,
        )
        result = run_sa(
            energy_fn=energy_fn_phase1,
            init_genome=init_genome,
            n_steps=n_steps_phase1,
            T0=T0_phase1,
            alpha=alpha_phase1,
            n_edges_range=n_edges_phase1,
            dfree_target=dfree_target,
            rng_seed=chain_seed,
            log_path=log_path,
            step_callback=step_callback_phase1,
            adaptive_restart=adaptive_restart_phase1,
        )
        phase1_results.append(result)
        best_e = result["best_energy"]
        best_str = f"{best_e:.4e}" if math.isfinite(best_e) else "inf"
        print(
            f"[Phase 1] Chain {i+1}/{n_chains} done. "
            f"best={best_str}  accepted={result['n_accepted']}  "
            f"rejected={result['n_rejected']}  invalid={result['n_invalid']}",
            flush=True,
        )

    # Pick the best Phase-1 genome (finite energy preferred; if all inf, take first)
    finite_results = [r for r in phase1_results if math.isfinite(r["best_energy"])]
    if finite_results:
        phase1_best = min(finite_results, key=lambda r: r["best_energy"])
    else:
        phase1_best = phase1_results[0]

    p1_best_e = phase1_best["best_energy"]
    print(
        f"[Phase 1] Best across {n_chains} chain(s): "
        f"energy={f'{p1_best_e:.4e}' if math.isfinite(p1_best_e) else 'inf'}",
        flush=True,
    )

    # ── Phase 2: single refinement chain ─────────────────────────────────────
    phase2_seed = int(rng.integers(0, 2**31 - 1))
    log_path_p2 = None
    if log_dir is not None:
        log_path_p2 = log_dir / f"log_phase2_seed{rng_seed}.npz"

    print(
        f"[Phase 2] Refinement: {n_steps_phase2} steps, T0={T0_phase2}, alpha={alpha_phase2}",
        flush=True,
    )
    phase2_result = run_sa(
        energy_fn=energy_fn_phase2,
        init_genome=phase1_best["best_genome"],
        n_steps=n_steps_phase2,
        T0=T0_phase2,
        alpha=alpha_phase2,
        n_edges_range=n_edges_phase2,
        dfree_target=dfree_target,
        rng_seed=phase2_seed,
        log_path=log_path_p2,
        step_callback=step_callback_phase2,
        adaptive_restart=adaptive_restart_phase2,
    )
    p2_best_e = phase2_result["best_energy"]
    print(
        f"[Phase 2] Done. best={f'{p2_best_e:.4e}' if math.isfinite(p2_best_e) else 'inf'}  "
        f"accepted={phase2_result['n_accepted']}  "
        f"rejected={phase2_result['n_rejected']}  invalid={phase2_result['n_invalid']}",
        flush=True,
    )

    return {
        **phase2_result,
        "phase1_best_energy": phase1_best["best_energy"],
        "phase1_results": phase1_results,
    }


if __name__ == "__main__":
    import time

    print("sa_search.py smoke test")
    rng = np.random.default_rng(0)

    print("Generating 3 random trellises from scratch...")
    t0 = time.perf_counter()
    genomes = [random_trellis_from_scratch(rng) for _ in range(3)]
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    from trellis_genome import nasa_k7_genome, genome_hash as ghash
    nasa_hash = ghash(nasa_k7_genome())
    for i, g in enumerate(genomes):
        h = ghash(g)
        assert h != nasa_hash, f"genome {i} has NASA hash"
        print(f"  genome {i}: hash={h[:12]}...  n_states={g['n_states']}")

    # Trivial energy function: sum of output_pair values (minimize → all zeros)
    def dummy_energy(genome: TrellisGenome, seed: int) -> float:
        return float(genome["output_pair"].sum())

    print("\nRunning single-chain SA on dummy energy (50 steps)...")
    result = run_sa(
        energy_fn=dummy_energy,
        init_genome=genomes[0],
        n_steps=50,
        T0=10.0,
        alpha=0.95,
        n_edges_range=(1, 3),
        dfree_target=1,
        rng_seed=42,
    )
    print(f"  best_energy={result['best_energy']:.1f}  accepted={result['n_accepted']}")
    print("PASS")
