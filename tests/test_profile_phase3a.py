from argparse import Namespace

import numpy as np

from profile_phase3a import (
    _component_rows,
    _format_component_table,
    _format_seed_table,
    _record_component,
    _serialize_profile_results,
    run_profile,
)


def test_component_aggregation_and_formatting():
    totals = {}
    counts = {}
    _record_component(totals, counts, "fitness_oracle_total", 0.5)
    _record_component(totals, counts, "fitness_oracle_total", 0.25)
    _record_component(totals, counts, "viterbi_decode", 0.75, count=3)
    rows = _component_rows(totals, counts)

    assert rows[0]["calls"] >= 1
    assert any(row["name"] == "fitness_oracle_total" for row in rows)
    assert any(row["name"] == "viterbi_decode" for row in rows)

    table = _format_component_table(rows)
    assert "component" in table
    assert "fitness_oracle_total" in table
    assert "viterbi_decode" in table


def test_profile_result_serialization_and_seed_table():
    results = {
        "config": {"n_seeds": 1, "pop_size": 4},
        "native": {"phase3_core": True, "viterbi_core": False},
        "e2e": {
            "seed_rows": [
                {
                    "seed": 0,
                    "init_time_s": 0.1,
                    "seed_time_s": 0.4,
                    "evaluation_time_s": 0.2,
                    "offspring_time_s": 0.1,
                    "n_generations": 2,
                }
            ],
            "generation_rows": [
                {
                    "seed": 0,
                    "generation": 0,
                    "evaluation_time_s": 0.1,
                    "offspring_time_s": 0.05,
                    "generation_time_s": 0.15,
                    "best_fitness": 0.1,
                    "mean_fitness": 0.2,
                    "plateau": 0,
                }
            ],
            "run_time_s": 0.4,
        },
        "components": {
            "rows": [
                {
                    "name": "fitness_oracle_total",
                    "calls": 2,
                    "total_s": 1.0,
                    "mean_ms": 500.0,
                    "share": 1.0,
                }
            ]
        },
    }

    payload = _serialize_profile_results(results)
    assert "component_name" in payload
    assert payload["e2e_seed"].shape == (1,)
    assert payload["gen_generation"].shape == (1,)

    table = _format_seed_table(results["e2e"]["seed_rows"])
    assert "seed | init_s" in table
    assert "0.1000" in table


def test_run_profile_smoke(tmp_path):
    args = Namespace(
        n_seeds=1,
        pop_size=4,
        n_generations=1,
        n_trials=5,
        dfree_target=8,
        output=tmp_path / "profile_phase3a.npz",
        component_repeats=1,
        snr_db=5.0,
        inr_db=5.0,
        skip_e2e=False,
        skip_components=False,
    )
    results = run_profile(args)
    payload = _serialize_profile_results(results)
    np.savez(args.output, **payload)

    assert args.output.exists()
    assert results["e2e"]["seed_rows"]
    assert results["components"]["rows"]
