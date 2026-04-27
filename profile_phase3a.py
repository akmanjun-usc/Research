from __future__ import annotations

import argparse
import functools
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np

import decoders
from channel import amplitude_from_inr, awgn_channel, generate_interference, noise_var_from_snr
from decoders import branch_metric_oracle, viterbi_decode
from eval import estimate_bler
from neural_bm import _encode_fixed_tail
from phase3_native import is_available as phase3_native_available, mutate_and_validate_native
from search import build_initial_population, run_ea
from trellis_genome import TrellisGenome, genome_to_trellis, nasa_k7_genome

RESULTS_DIR = Path("results/phase3a")
DEFAULT_OUTPUT = RESULTS_DIR / "profile_phase3a.npz"


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Phase 3a end-to-end and component timings.")
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--pop-size", type=int, default=10)
    parser.add_argument("--n-generations", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--dfree-target", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--component-repeats", type=int, default=3)
    parser.add_argument("--snr-db", type=float, default=5.0)
    parser.add_argument("--inr-db", type=float, default=5.0)
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--skip-components", action="store_true")
    return parser


def _record_component(
    component_totals: dict[str, float],
    component_counts: dict[str, int],
    name: str,
    elapsed_s: float,
    count: int = 1,
) -> None:
    component_totals[name] = float(component_totals.get(name, 0.0)) + float(elapsed_s)
    component_counts[name] = int(component_counts.get(name, 0)) + int(count)


def _component_rows(component_totals: dict[str, float], component_counts: dict[str, int]) -> list[dict]:
    rows: list[dict] = []
    total_measured = float(sum(component_totals.values()))
    for name in sorted(component_totals):
        total_s = float(component_totals[name])
        calls = int(component_counts[name])
        mean_ms = 1000.0 * total_s / calls if calls > 0 else 0.0
        share = total_s / total_measured if total_measured > 0 else 0.0
        rows.append(
            {
                "name": name,
                "calls": calls,
                "total_s": total_s,
                "mean_ms": mean_ms,
                "share": share,
            }
        )
    return rows


def _format_component_table(rows: list[dict]) -> str:
    lines = [
        "component                  | calls | total_s  | mean_ms | share",
        "---------------------------+-------+----------+---------+-------",
    ]
    for row in rows:
        lines.append(
            f"{row['name']:<26} | "
            f"{row['calls']:>5d} | "
            f"{row['total_s']:>8.4f} | "
            f"{row['mean_ms']:>7.2f} | "
            f"{100.0 * row['share']:>5.1f}%"
        )
    return "\n".join(lines)


def _format_seed_table(seed_rows: list[dict]) -> str:
    lines = [
        "seed | init_s  | seed_s  | eval_s  | offspring_s | generations",
        "-----+---------+---------+---------+-------------+------------",
    ]
    for row in seed_rows:
        lines.append(
            f"{row['seed']:>4d} | "
            f"{row['init_time_s']:>7.4f} | "
            f"{row['seed_time_s']:>7.4f} | "
            f"{row['evaluation_time_s']:>7.4f} | "
            f"{row['offspring_time_s']:>11.4f} | "
            f"{row['n_generations']:>10d}"
        )
    return "\n".join(lines)


def _serialize_profile_results(results: dict) -> dict[str, np.ndarray]:
    seed_rows = results["e2e"]["seed_rows"]
    generation_rows = results["e2e"]["generation_rows"]
    eval_component_rows = results["e2e"]["evaluation_component_rows"]
    component_rows = results["components"]["rows"]
    config = results["config"]
    return {
        "config_keys": np.array(sorted(config.keys()), dtype=object),
        "config_values": np.array([str(config[key]) for key in sorted(config.keys())], dtype=object),
        "native_phase3_core": np.array([bool(results["native"]["phase3_core"])], dtype=bool),
        "native_viterbi_core": np.array([bool(results["native"]["viterbi_core"])], dtype=bool),
        "e2e_seed": np.array([row["seed"] for row in seed_rows], dtype=np.int32),
        "e2e_init_time_s": np.array([row["init_time_s"] for row in seed_rows], dtype=np.float64),
        "e2e_seed_time_s": np.array([row["seed_time_s"] for row in seed_rows], dtype=np.float64),
        "e2e_eval_time_s": np.array([row["evaluation_time_s"] for row in seed_rows], dtype=np.float64),
        "e2e_offspring_time_s": np.array([row["offspring_time_s"] for row in seed_rows], dtype=np.float64),
        "e2e_generations": np.array([row["n_generations"] for row in seed_rows], dtype=np.int32),
        "gen_seed": np.array([row["seed"] for row in generation_rows], dtype=np.int32),
        "gen_generation": np.array([row["generation"] for row in generation_rows], dtype=np.int32),
        "gen_eval_time_s": np.array([row["evaluation_time_s"] for row in generation_rows], dtype=np.float64),
        "gen_offspring_time_s": np.array([row["offspring_time_s"] for row in generation_rows], dtype=np.float64),
        "gen_total_time_s": np.array([row["generation_time_s"] for row in generation_rows], dtype=np.float64),
        "gen_best_fitness": np.array([row["best_fitness"] for row in generation_rows], dtype=np.float64),
        "gen_mean_fitness": np.array([row["mean_fitness"] for row in generation_rows], dtype=np.float64),
        "gen_plateau": np.array([row["plateau"] for row in generation_rows], dtype=np.int32),
        "eval_component_name": np.array([row["name"] for row in eval_component_rows], dtype=object),
        "eval_component_calls": np.array([row["calls"] for row in eval_component_rows], dtype=np.int32),
        "eval_component_total_s": np.array([row["total_s"] for row in eval_component_rows], dtype=np.float64),
        "eval_component_mean_ms": np.array([row["mean_ms"] for row in eval_component_rows], dtype=np.float64),
        "eval_component_share": np.array([row["share"] for row in eval_component_rows], dtype=np.float64),
        "component_name": np.array([row["name"] for row in component_rows], dtype=object),
        "component_calls": np.array([row["calls"] for row in component_rows], dtype=np.int32),
        "component_total_s": np.array([row["total_s"] for row in component_rows], dtype=np.float64),
        "component_mean_ms": np.array([row["mean_ms"] for row in component_rows], dtype=np.float64),
        "component_share": np.array([row["share"] for row in component_rows], dtype=np.float64),
    }


def _profile_fitness_oracle_call(
    genome: TrellisGenome,
    seed: int,
    n_trials: int,
    snr_db: float,
    inr_db: float,
    component_totals: dict[str, float],
    component_counts: dict[str, int],
) -> float:
    t0 = perf_counter()
    trellis = genome_to_trellis(genome)
    _record_component(component_totals, component_counts, "genome_to_trellis", perf_counter() - t0)

    def encode_fn(info_bits):
        start = perf_counter()
        coded = _encode_fixed_tail(info_bits, trellis)
        _record_component(component_totals, component_counts, "encode_fixed_tail", perf_counter() - start)
        return coded

    def channel_fn(symbols, snr_db, inr_db, period, phase, rng):
        start = perf_counter()
        received = awgn_channel(symbols, snr_db, inr_db, period, phase, rng)
        _record_component(component_totals, component_counts, "awgn_channel", perf_counter() - start)
        return received

    def decode_fn(received, period, phase, snr_db, inr_db):
        nv = noise_var_from_snr(snr_db)
        amp = amplitude_from_inr(inr_db, nv)
        start = perf_counter()
        interference = generate_interference(len(received), amp, period, phase)
        _record_component(component_totals, component_counts, "generate_interference", perf_counter() - start)

        start = perf_counter()
        decoded = viterbi_decode(
            received.reshape(-1, 2),
            trellis,
            branch_metric_oracle,
            noise_var=nv,
            interference=interference,
        )
        _record_component(component_totals, component_counts, "viterbi_decode", perf_counter() - start)
        return decoded

    start = perf_counter()
    result = estimate_bler(
        encode_fn,
        decode_fn,
        channel_fn,
        snr_db=snr_db,
        inr_db=inr_db,
        n_trials=n_trials,
        seed=seed,
    )
    _record_component(component_totals, component_counts, "fitness_oracle_total", perf_counter() - start)
    return float(result["bler"])


def _run_component_profile(args) -> dict:
    component_totals: dict[str, float] = defaultdict(float)
    component_counts: dict[str, int] = defaultdict(int)
    base_genome = nasa_k7_genome()
    rng = np.random.default_rng(0)

    start = perf_counter()
    build_initial_population(
        pop_size=args.pop_size,
        base_genome=base_genome,
        dfree_target=args.dfree_target,
        rng=rng,
    )
    _record_component(component_totals, component_counts, "build_initial_population", perf_counter() - start)

    for rep in range(args.component_repeats):
        start = perf_counter()
        result = mutate_and_validate_native(
            base_genome,
            n_edges=2,
            max_attempts=1000,
            dfree_target=args.dfree_target,
            seed=rep,
        )
        _record_component(component_totals, component_counts, "mutate_and_validate_native", perf_counter() - start)
        if result is None:
            raise RuntimeError("mutate_and_validate_native failed during profiling")

    for rep in range(args.component_repeats):
        _profile_fitness_oracle_call(
            genome=base_genome,
            seed=rep,
            n_trials=args.n_trials,
            snr_db=args.snr_db,
            inr_db=args.inr_db,
            component_totals=component_totals,
            component_counts=component_counts,
        )

    rows = _component_rows(component_totals, component_counts)
    return {"rows": rows}


def _run_e2e_profile(args) -> dict:
    seed_rows: list[dict] = []
    generation_rows: list[dict] = []
    evaluation_component_totals: dict[str, float] = defaultdict(float)
    evaluation_component_counts: dict[str, int] = defaultdict(int)
    run_start = perf_counter()

    for rng_seed in range(args.n_seeds):
        seed_start = perf_counter()
        rng = np.random.default_rng(rng_seed)

        init_start = perf_counter()
        init_pop = build_initial_population(
            pop_size=args.pop_size,
            base_genome=nasa_k7_genome(),
            dfree_target=args.dfree_target,
            rng=rng,
        )
        init_time_s = perf_counter() - init_start

        fitness_fn = functools.partial(
            _profile_fitness_oracle_call,
            n_trials=args.n_trials,
            snr_db=args.snr_db,
            inr_db=args.inr_db,
            component_totals=evaluation_component_totals,
            component_counts=evaluation_component_counts,
        )

        seed_generation_rows: list[dict] = []

        def generation_callback(snapshot: dict) -> None:
            seed_generation_rows.append(snapshot.copy())

        run_ea(
            fitness_fn=fitness_fn,
            init_population=init_pop,
            n_generations=args.n_generations,
            pop_size=args.pop_size,
            elite_size=2,
            dfree_target=args.dfree_target,
            rng_seed=rng_seed,
            generation_callback=generation_callback,
        )

        seed_time_s = perf_counter() - seed_start
        evaluation_time_s = float(sum(row["evaluation_time_s"] for row in seed_generation_rows))
        offspring_time_s = float(sum(row["offspring_time_s"] for row in seed_generation_rows))
        seed_rows.append(
            {
                "seed": int(rng_seed),
                "init_time_s": float(init_time_s),
                "seed_time_s": float(seed_time_s),
                "evaluation_time_s": evaluation_time_s,
                "offspring_time_s": offspring_time_s,
                "n_generations": int(len(seed_generation_rows)),
            }
        )
        generation_rows.extend(seed_generation_rows)

    return {
        "seed_rows": seed_rows,
        "generation_rows": generation_rows,
        "evaluation_component_rows": _component_rows(
            evaluation_component_totals,
            evaluation_component_counts,
        ),
        "run_time_s": float(perf_counter() - run_start),
    }


def _native_status() -> dict[str, bool]:
    return {
        "phase3_core": bool(phase3_native_available()),
        "viterbi_core": bool(decoders._load_c_lib() is not None),
    }


def run_profile(args) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "n_seeds": args.n_seeds,
            "pop_size": args.pop_size,
            "n_generations": args.n_generations,
            "n_trials": args.n_trials,
            "dfree_target": args.dfree_target,
            "component_repeats": args.component_repeats,
            "snr_db": args.snr_db,
            "inr_db": args.inr_db,
        },
        "native": _native_status(),
        "e2e": {
            "seed_rows": [],
            "generation_rows": [],
            "evaluation_component_rows": [],
            "run_time_s": 0.0,
        },
        "components": {"rows": []},
    }

    if not args.skip_e2e:
        results["e2e"] = _run_e2e_profile(args)
    if not args.skip_components:
        results["components"] = _run_component_profile(args)

    return results


def _print_summary(results: dict) -> None:
    config = results["config"]
    print(
        "Phase 3a profiler "
        f"(seeds={config['n_seeds']}, pop={config['pop_size']}, gens={config['n_generations']}, "
        f"trials={config['n_trials']}, dfree={config['dfree_target']})"
    )
    print(
        f"Native availability: phase3_core={'yes' if results['native']['phase3_core'] else 'no'}, "
        f"viterbi_core={'yes' if results['native']['viterbi_core'] else 'no'}"
    )
    if results["e2e"]["seed_rows"]:
        print()
        print(_format_seed_table(results["e2e"]["seed_rows"]))
        print(f"total_run_s: {results['e2e']['run_time_s']:.4f}")
    if results["e2e"]["evaluation_component_rows"]:
        print()
        print("EA evaluation breakdown:")
        print(_format_component_table(results["e2e"]["evaluation_component_rows"]))
    if results["components"]["rows"]:
        print()
        print("Component microbenchmarks:")
        print(_format_component_table(results["components"]["rows"]))


def main(argv: Optional[list[str]] = None) -> None:
    args = _make_parser().parse_args(argv)
    results = run_profile(args)
    payload = _serialize_profile_results(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    _print_summary(results)
    print(f"\nSaved profiling results to: {args.output}")


if __name__ == "__main__":
    main()
