import numpy as np

from search import build_initial_population, run_ea
from trellis_genome import nasa_k7_genome


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
