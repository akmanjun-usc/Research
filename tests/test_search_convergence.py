import numpy as np

from search import build_initial_population, run_ea
from trellis_genome import nasa_k7_genome


def test_run_ea_progress_callback_reports_monotone_best():
    target = nasa_k7_genome()
    snapshots: list[dict] = []

    def hamming_to_nasa_k7(genome, _seed):
        return float(
            np.count_nonzero(genome["next_state"] != target["next_state"])
            + np.count_nonzero(genome["output_pair"] != target["output_pair"])
        )

    rng = np.random.default_rng(6)
    init_pop = build_initial_population(
        pop_size=20,
        base_genome=target,
        dfree_target=8,
        rng=rng,
    )

    result = run_ea(
        fitness_fn=hamming_to_nasa_k7,
        init_population=init_pop,
        n_generations=100,
        pop_size=20,
        elite_size=2,
        n_edges_range=(1, 3),
        plateau_patience=50,
        rng_seed=6,
        progress_callback=lambda snapshot: snapshots.append(snapshot.copy()),
    )

    assert snapshots
    assert snapshots[-1]["is_complete"] is True
    assert all("seed" in snapshot for snapshot in snapshots)
    assert all("generation" in snapshot for snapshot in snapshots)
    assert all("best_fitness" in snapshot for snapshot in snapshots)
    assert all("mean_fitness" in snapshot for snapshot in snapshots)
    assert all("plateau" in snapshot for snapshot in snapshots)
    assert all("convergence_generation" in snapshot for snapshot in snapshots)
    best_values = [snapshot["best_fitness"] for snapshot in snapshots if not snapshot["is_complete"]]
    assert best_values == sorted(best_values, reverse=True)
    assert snapshots[-1]["best_fitness"] == result["best_fitness"]


def test_ea_converges_on_synthetic_fitness():
    target = nasa_k7_genome()

    def hamming_to_nasa_k7(genome, _seed):
        return float(
            np.count_nonzero(genome["next_state"] != target["next_state"])
            + np.count_nonzero(genome["output_pair"] != target["output_pair"])
        )

    rng = np.random.default_rng(5)
    init_pop = build_initial_population(
        pop_size=20,
        base_genome=target,
        dfree_target=8,
        rng=rng,
    )
    result = run_ea(
        fitness_fn=hamming_to_nasa_k7,
        init_population=init_pop,
        n_generations=100,
        pop_size=20,
        elite_size=2,
        n_edges_range=(1, 3),
        plateau_patience=50,
        rng_seed=5,
    )
    assert result["best_fitness"] == 0
    assert result["convergence_generation"] <= 50
