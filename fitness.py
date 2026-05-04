"""
fitness.py — Phase 3 fitness functions
"""

from __future__ import annotations

import numpy as np
import torch

from channel import (
    amplitude_from_inr, awgn_channel, bpsk_modulate,
    generate_interference, noise_var_from_snr,
    K_INFO,
)
from decoders import branch_metric_oracle, viterbi_decode
from eval import estimate_bler
from neural_bm import _encode_fixed_tail
from trellis_genome import TrellisGenome, genome_to_trellis


def fitness_oracle(
    genome: TrellisGenome,
    seed: int,
    n_trials: int = 1000,
    snr_db: float = 5.0,
    inr_db: float = 5.0,
    early_stop_errors: int = 100,
) -> float:
    trellis = genome_to_trellis(genome)

    def encode_fn(info_bits):
        return bpsk_modulate(_encode_fixed_tail(info_bits, trellis))

    def decode_fn(received, period, phase, snr_db, inr_db):
        nv = noise_var_from_snr(snr_db)
        amp = amplitude_from_inr(inr_db, nv)
        interf = generate_interference(len(received), amp, period, phase)
        return viterbi_decode(
            received.reshape(-1, 2),
            trellis,
            branch_metric_oracle,
            noise_var=nv,
            interference=interf,
        )

    result = estimate_bler(
        encode_fn,
        decode_fn,
        awgn_channel,
        snr_db=snr_db,
        inr_db=inr_db,
        n_trials=n_trials,
        seed=seed,
        early_stop_errors=early_stop_errors,
    )
    return float(result["bler"])


def fitness_n2(
    genome: TrellisGenome,
    seed: int,
    n_trials: int = 1000,
    snr_db: float = 5.0,
    inr_db: float = 5.0,
    model=None,
    device: str = "cpu",
) -> float:
    """
    Batched N2 fitness: runs all n_trials through the GRU in one forward pass
    instead of n_trials separate calls, giving ~100x speedup on CPU.
    """
    from neural_bm import build_branch_output_index, pair_received_signal
    from phase3_native import viterbi_neural_bm_native

    trellis = genome_to_trellis(genome)
    index_table = build_branch_output_index(trellis)
    S = trellis.n_states

    rng = np.random.default_rng(seed)
    noise_var = noise_var_from_snr(snr_db)
    amp = amplitude_from_inr(inr_db, noise_var)

    # Generate all trials: encode + channel
    info_bits_list = []
    received_list = []
    for _ in range(n_trials):
        info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
        period = rng.integers(8, 33)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        coded = bpsk_modulate(_encode_fixed_tail(info_bits, trellis))
        noise = rng.standard_normal(len(coded)) * np.sqrt(noise_var)
        interf = generate_interference(len(coded), amp, period, phase)
        info_bits_list.append(info_bits)
        received_list.append(coded + noise + interf)

    # Batch NN inference: one forward pass for all trials
    received_arr = np.stack(received_list)             # (n_trials, N)
    paired = pair_received_signal(received_arr)        # (n_trials, T, 2)
    dev = torch.device(device)
    x = torch.tensor(paired, dtype=torch.float32, device=dev)
    with torch.no_grad():
        nn_out = model(x)                              # (n_trials, T, 4)
    bm_batch = nn_out.cpu().numpy() / (2.0 * float(noise_var))  # (n_trials, T, 4)
    T = bm_batch.shape[1]

    # Viterbi per trial using C extension (sequential — DP is inherently serial)
    n_errors = 0
    for i in range(n_trials):
        decoded = viterbi_neural_bm_native(
            bm_batch[i], trellis.next_state, index_table, S, T, K_INFO
        )
        if not np.array_equal(info_bits_list[i], decoded):
            n_errors += 1

    return float(n_errors / n_trials)
