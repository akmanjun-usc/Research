"""
fitness.py — Phase 3 fitness functions
"""

from __future__ import annotations

from channel import amplitude_from_inr, awgn_channel, bpsk_modulate, generate_interference, noise_var_from_snr
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
    raise NotImplementedError("fitness_n2 is implemented in Phase 3b")
