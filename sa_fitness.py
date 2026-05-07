"""
sa_fitness.py — Cascaded fitness functions and BM alignment energy for Phase 3c SA
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from channel import (
    K_INFO,
    amplitude_from_inr,
    bpsk_modulate,
    generate_interference,
    noise_var_from_snr,
)
from fitness import fitness_n2, fitness_oracle
from neural_bm import (
    NeuralBranchMetric,
    _encode_fixed_tail,
    compute_oracle_metrics,
    pair_received_signal,
)
from trellis_genome import TrellisGenome, genome_to_trellis

REJECTED_ENERGY: float = float("inf")


def _derive_seeds(seed: int) -> tuple[int, int, int]:
    """Derive three independent child seeds from a root seed (deterministic)."""
    rng = np.random.default_rng(seed)
    return tuple(int(rng.integers(0, 2**31 - 1)) for _ in range(3))  # type: ignore[return-value]


def cascaded_fitness_oracle(
    genome: TrellisGenome,
    seed: int,
    stage1_snr_db: float = 10.0,
    stage1_inr_db: float = 5.0,
    stage1_n_trials: int = 200,
    stage1_bler_threshold: float = 1e-4,
    stage1_early_stop: int = 5,
    stage2_snr_db: float = 6.0,
    stage2_inr_db: float = 5.0,
    stage2_n_trials: int = 500,
    stage2_bler_threshold: float = 0.5,
    stage2_early_stop: int = 50,
    stage3_snr_db: float = 3.0,
    stage3_inr_db: float = 5.0,
    stage3_n_trials: int = 500,
) -> float:
    """
    Three-stage cascaded fitness using oracle Viterbi decoder.

    Stage 1: SNR=10 dB, 200 trials — cheap fast reject (BLER > 1e-4 → inf)
    Stage 2: SNR=6  dB, 500 trials — mid  reject  (BLER > 0.5  → inf)
    Stage 3: SNR=3  dB, 500 trials — actual energy returned to SA

    Stage 2 threshold is intentionally loose (0.5) so that random starting
    trellises are not all rejected — only codes with no coding gain at 6 dB
    are filtered. Tighten once SA has found competitive candidates.

    Seeds are derived deterministically from `seed` so the same
    (genome, seed) pair always returns the same value.
    """
    s1, s2, s3 = _derive_seeds(seed)

    bler1 = fitness_oracle(
        genome, s1,
        n_trials=stage1_n_trials,
        snr_db=stage1_snr_db,
        inr_db=stage1_inr_db,
        early_stop_errors=stage1_early_stop,
    )
    if bler1 > stage1_bler_threshold:
        return REJECTED_ENERGY

    bler2 = fitness_oracle(
        genome, s2,
        n_trials=stage2_n_trials,
        snr_db=stage2_snr_db,
        inr_db=stage2_inr_db,
        early_stop_errors=stage2_early_stop,
    )
    if bler2 > stage2_bler_threshold:
        return REJECTED_ENERGY

    bler3 = fitness_oracle(
        genome, s3,
        n_trials=stage3_n_trials,
        snr_db=stage3_snr_db,
        inr_db=stage3_inr_db,
        early_stop_errors=stage3_n_trials,
    )
    return bler3


def cascaded_fitness_n2(
    genome: TrellisGenome,
    seed: int,
    model: NeuralBranchMetric,
    device: str = "cpu",
    stage1_snr_db: float = 10.0,
    stage1_inr_db: float = 5.0,
    stage1_n_trials: int = 200,
    stage1_bler_threshold: float = 1e-4,
    stage2_snr_db: float = 6.0,
    stage2_inr_db: float = 5.0,
    stage2_n_trials: int = 500,
    stage2_bler_threshold: float = 0.5,
    stage3_snr_db: float = 3.0,
    stage3_inr_db: float = 5.0,
    stage3_n_trials: int = 500,
) -> float:
    """
    Three-stage cascaded fitness using N2 neural branch metric decoder.
    Same structure as cascaded_fitness_oracle; wraps fitness_n2 instead.
    Stage 2 threshold set to 0.5 for the same reason as the oracle version.
    """
    s1, s2, s3 = _derive_seeds(seed)

    bler1 = fitness_n2(
        genome, s1,
        n_trials=stage1_n_trials,
        snr_db=stage1_snr_db,
        inr_db=stage1_inr_db,
        model=model,
        device=device,
    )
    if bler1 > stage1_bler_threshold:
        return REJECTED_ENERGY

    bler2 = fitness_n2(
        genome, s2,
        n_trials=stage2_n_trials,
        snr_db=stage2_snr_db,
        inr_db=stage2_inr_db,
        model=model,
        device=device,
    )
    if bler2 > stage2_bler_threshold:
        return REJECTED_ENERGY

    bler3 = fitness_n2(
        genome, s3,
        n_trials=stage3_n_trials,
        snr_db=stage3_snr_db,
        inr_db=stage3_inr_db,
        model=model,
        device=device,
    )
    return bler3


def combined_energy(
    genome: TrellisGenome,
    seed: int,
    model: NeuralBranchMetric,
    device: str = "cpu",
    n_align: int = 100,
    snr_db: float = 5.0,
    inr_db: float = 5.0,
) -> float:
    """
    Pure BM alignment energy — no BLER simulation required.

    Measures normalized MSE between oracle branch metrics and N2 branch metrics
    over n_align random received signals for this genome's trellis.

    oracle BM[b,t,j] = −‖y[b,t] − h_j − interf[b,t]‖²  (all 4 BPSK hypotheses)
    N2 BM[b,t,j]     = GRU(y)[b,t,j]

    Both are normalized to zero-mean unit-variance per time step across the 4
    hypotheses before computing MSE, removing scale differences.

    Lower alignment_gap → N2 agrees with oracle → better N2 decoding BLER.
    The seed controls which random signals are used, ensuring reproducibility.
    """
    trellis = genome_to_trellis(genome)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    noise_var = noise_var_from_snr(snr_db)
    amp = amplitude_from_inr(inr_db, noise_var)

    y_list = []
    interf_list = []
    for _ in range(n_align):
        info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
        period = float(rng.integers(8, 33))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))

        coded = _encode_fixed_tail(info_bits, trellis)
        symbols = bpsk_modulate(coded)
        N = len(symbols)

        noise = rng.standard_normal(N) * np.sqrt(noise_var)
        interf = generate_interference(N, amp, period, phase)
        y_list.append(symbols + noise + interf)
        interf_list.append(interf)

    y_arr = np.stack(y_list)           # (n_align, N)
    interf_arr = np.stack(interf_list)  # (n_align, N)

    y_paired = pair_received_signal(y_arr)       # (n_align, T, 2)
    i_paired = pair_received_signal(interf_arr)  # (n_align, T, 2)

    y_t = torch.tensor(y_paired, dtype=torch.float32, device=dev)
    i_t = torch.tensor(i_paired, dtype=torch.float32, device=dev)

    with torch.no_grad():
        bm_oracle = compute_oracle_metrics(y_t, i_t)  # (n_align, T, 4)
        bm_n2 = model(y_t)                             # (n_align, T, 4)

    eps = 1e-8
    # Normalize per (batch, time-step) across the 4 hypotheses
    bm_oracle_norm = (bm_oracle - bm_oracle.mean(-1, keepdim=True)) / (
        bm_oracle.std(-1, keepdim=True).clamp(min=eps)
    )
    bm_n2_norm = (bm_n2 - bm_n2.mean(-1, keepdim=True)) / (
        bm_n2.std(-1, keepdim=True).clamp(min=eps)
    )

    alignment_gap = float(((bm_oracle_norm - bm_n2_norm) ** 2).mean().item())
    return alignment_gap


if __name__ == "__main__":
    import time

    print("sa_fitness.py smoke test")
    from trellis_genome import nasa_k7_genome

    nasa = nasa_k7_genome()

    # cascaded_fitness_oracle: NASA K=7 should not be rejected
    print("Testing cascaded_fitness_oracle on NASA K=7...")
    t0 = time.perf_counter()
    val = cascaded_fitness_oracle(nasa, seed=0)
    print(f"  BLER={val:.4e}  ({time.perf_counter()-t0:.1f}s)")
    assert math.isfinite(val), "NASA K=7 should not be rejected by cascaded_fitness_oracle"
    print("  PASS (finite)")

    # combined_energy: requires model
    ckpt_path = "results/phase2b/checkpoints/best_model_seed32.pt"
    import pathlib
    if pathlib.Path(ckpt_path).exists():
        from neural_bm import load_model
        model, _ = load_model(ckpt_path)
        model.eval()
        print("\nTesting combined_energy on NASA K=7...")
        t0 = time.perf_counter()
        gap = combined_energy(nasa, seed=0, model=model, device="cpu", n_align=20)
        print(f"  alignment_gap={gap:.6f}  ({time.perf_counter()-t0:.1f}s)")
        assert math.isfinite(gap) and gap >= 0.0, "alignment_gap must be finite and non-negative"
        print("  PASS")
    else:
        print(f"\n(Skipping combined_energy test: {ckpt_path} not found)")

    print("\nAll sa_fitness smoke tests PASSED")
