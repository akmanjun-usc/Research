from run_phase3a import (
    _blank_seed_status,
    _render_progress_table,
    _should_print_progress,
    _update_seed_status,
)


def test_progress_table_pending_running_done_rendering():
    seed_statuses = [_blank_seed_status(seed) for seed in range(5)]
    _update_seed_status(
        seed_statuses,
        {
            "seed": 0,
            "generation": 9,
            "best_fitness": 1.2e-2,
            "mean_fitness": 2.5e-2,
            "plateau": 3,
            "is_complete": False,
            "convergence_generation": 7,
        },
    )
    _update_seed_status(
        seed_statuses,
        {
            "seed": 1,
            "generation": 17,
            "best_fitness": 9.0e-3,
            "mean_fitness": 1.7e-2,
            "plateau": 0,
            "is_complete": True,
            "convergence_generation": 17,
        },
    )

    table = _render_progress_table(seed_statuses)
    assert "seed | status" in table
    assert "0 | running" in table
    assert "1 | done" in table
    assert "2 | pending" in table
    assert "1.2000e-02" in table
    assert "9.0000e-03" in table
    assert "--" in table


def test_should_print_progress_every_ten_and_completion():
    assert _should_print_progress({"generation": 0, "is_complete": False}, every=10) is True
    assert _should_print_progress({"generation": 8, "is_complete": False}, every=10) is False
    assert _should_print_progress({"generation": 9, "is_complete": False}, every=10) is True
    assert _should_print_progress({"generation": 19, "is_complete": False}, every=10) is True
    assert _should_print_progress({"generation": 3, "is_complete": True}, every=10) is True
