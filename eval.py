"""
eval.py — Monte Carlo BLER estimation, SNR/INR sweeps

Part of: EE597 Search-Designed Trellis Codes Project
"""

from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from scipy.special import erfc
from channel import (
    bpsk_modulate, awgn_channel, noise_var_from_snr,
    amplitude_from_inr, generate_interference, K_INFO, N_CODED,
)
from trellis import load_nasa_k7, Trellis
from decoders import viterbi_decode, branch_metric_mismatched, branch_metric_oracle
from interference_est import estimate_and_cancel
from plot_utils import plot_bler_vs_snr, plot_bler_vs_inr
from compute_cost import (
    count_flops_viterbi, count_flops_ic_viterbi,
    measure_latency_ms, make_compute_table, assert_within_budget,
    N_STATES,
)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"


# ─────────────────────────────────────────────
# Reliability check
# ─────────────────────────────────────────────
def is_reliable(result: dict, min_errors: int = 50) -> bool:
    """BLER estimates with < min_errors errors are unreliable."""
    return result['n_errors'] >= min_errors


# ─────────────────────────────────────────────
# Core BLER estimation
# ─────────────────────────────────────────────
def estimate_bler(
    encode_fn: callable,
    decode_fn: callable,
    channel_fn: callable,
    snr_db: float,
    inr_db: float,
    n_trials: int,
    seed: int = 0,
    period_range: tuple[int, int] = (8, 32),
) -> dict:
    """
    Monte Carlo BLER estimation.

    Args:
        encode_fn: info_bits -> coded_symbols (BPSK)
        decode_fn: received -> decoded_info_bits
        channel_fn: (symbols, snr_db, inr_db, period, phase, rng) -> received
        snr_db: signal-to-noise ratio in dB
        inr_db: interference-to-noise ratio in dB
        n_trials: number of Monte Carlo trials
        seed: random seed
        period_range: range of interference periods

    Returns:
        dict with bler, n_errors, n_trials, ci_95, snr_db, inr_db
    """
    rng = np.random.default_rng(seed)
    n_errors = 0

    for _ in range(n_trials):
        info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)

        # Random interference parameters
        period = rng.integers(period_range[0], period_range[1] + 1)
        phase = rng.uniform(0, 2 * np.pi)

        # Encode and modulate
        coded_symbols = encode_fn(info_bits)

        # Channel
        received = channel_fn(coded_symbols, snr_db, inr_db, period, phase, rng)

        # Decode
        decoded = decode_fn(received, period, phase, snr_db, inr_db)

        if not np.array_equal(info_bits, decoded):
            n_errors += 1

    bler = n_errors / n_trials
    ci = 1.96 * np.sqrt(bler * (1 - bler) / n_trials) if n_trials > 0 else 0.0

    return dict(
        bler=bler, n_errors=n_errors, n_trials=n_trials,
        ci_95=ci, snr_db=snr_db, inr_db=inr_db,
    )


# ─────────────────────────────────────────────
# Method wrapper factories
# ─────────────────────────────────────────────
def make_encoder(trellis: Trellis) -> callable:
    """Returns encoder function: info_bits -> BPSK symbols."""
    def encode_fn(info_bits: np.ndarray) -> np.ndarray:
        coded = trellis.encode(info_bits)
        return bpsk_modulate(coded)
    return encode_fn


def make_decoder_b1(trellis: Trellis) -> callable:
    """B1: Mismatched Viterbi (ignores interference)."""
    def decode_fn(received: np.ndarray, period: float, phase: float,
                  snr_db: float, inr_db: float) -> np.ndarray:
        nv = noise_var_from_snr(snr_db)
        rx = received.reshape(-1, 2)
        return viterbi_decode(rx, trellis, branch_metric_mismatched, noise_var=nv)
    return decode_fn


def make_decoder_b2(trellis: Trellis) -> callable:
    """B2: Oracle Viterbi (knows interference perfectly)."""
    def decode_fn(received: np.ndarray, period: float, phase: float,
                  snr_db: float, inr_db: float) -> np.ndarray:
        nv = noise_var_from_snr(snr_db)
        amp = amplitude_from_inr(inr_db, nv)
        interf = generate_interference(len(received), amp, period, phase)
        rx = received.reshape(-1, 2)
        return viterbi_decode(rx, trellis, branch_metric_oracle,
                              noise_var=nv, interference=interf)
    return decode_fn


def make_decoder_b5(trellis: Trellis) -> callable:
    """B5: Interference cancellation + Viterbi."""
    def decode_fn(received: np.ndarray, period: float, phase: float,
                  snr_db: float, inr_db: float) -> np.ndarray:
        nv = noise_var_from_snr(snr_db)
        cleaned, _ = estimate_and_cancel(received, period_range=(8, 32))
        rx = cleaned.reshape(-1, 2)
        return viterbi_decode(rx, trellis, branch_metric_mismatched, noise_var=nv)
    return decode_fn


# ─────────────────────────────────────────────
# SNR sweep
# ─────────────────────────────────────────────
def sweep_snr(
    methods: dict[str, tuple[callable, callable]],
    snr_range: np.ndarray,
    inr_db: float,
    n_trials: int,
    results_dir: Path,
    tag: str = "exp",
    seed: int = 0,
) -> dict[str, list[dict]]:
    """
    Sweep SNR for all methods. Saves results after each point.

    Args:
        methods: {name: (encode_fn, decode_fn)}
        snr_range: array of SNR values in dB
        inr_db: fixed INR in dB
        n_trials: Monte Carlo trials per point
        results_dir: where to save .npz results
        tag: experiment tag for filenames
        seed: base random seed

    Returns:
        {method_name: list of estimate_bler results}
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {name: [] for name in methods}

    for snr_db in snr_range:
        for name, (encode_fn, decode_fn) in methods.items():
            r = estimate_bler(
                encode_fn, decode_fn, awgn_channel,
                snr_db=float(snr_db), inr_db=inr_db,
                n_trials=n_trials, seed=seed,
            )
            results[name].append(r)

            # Save immediately after each data point
            _save_method_results(results[name], name, inr_db, results_dir, tag, seed)

            reliable = "OK" if is_reliable(r) else "LOW"
            print(f"  [{name}] SNR={snr_db:.1f}dB -> BLER={r['bler']:.2e} "
                  f"({r['n_errors']}/{r['n_trials']} errors) [{reliable}]")

    return results


# ─────────────────────────────────────────────
# INR sweep
# ─────────────────────────────────────────────
def sweep_inr(
    methods: dict[str, tuple[callable, callable]],
    inr_range: np.ndarray,
    snr_db: float,
    n_trials: int,
    results_dir: Path,
    tag: str = "exp",
    seed: int = 0,
) -> dict[str, list[dict]]:
    """
    Sweep INR for all methods at fixed SNR. Saves results after each point.

    Args:
        methods: {name: (encode_fn, decode_fn)}
        inr_range: array of INR values in dB
        snr_db: fixed SNR in dB
        n_trials: Monte Carlo trials per point
        results_dir: where to save .npz results
        tag: experiment tag for filenames
        seed: base random seed

    Returns:
        {method_name: list of estimate_bler results}
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {name: [] for name in methods}

    for inr_db in inr_range:
        for name, (encode_fn, decode_fn) in methods.items():
            r = estimate_bler(
                encode_fn, decode_fn, awgn_channel,
                snr_db=snr_db, inr_db=float(inr_db),
                n_trials=n_trials, seed=seed,
            )
            results[name].append(r)

            # Save immediately after each data point
            save_path = results_dir / f"{tag}_{name}_snr{snr_db:.0f}dB_inr_sweep.npz"
            np.savez(
                save_path,
                inr_db=np.array([p['inr_db'] for p in results[name]]),
                bler=np.array([p['bler'] for p in results[name]]),
                ci_95=np.array([p['ci_95'] for p in results[name]]),
                n_errors=np.array([p['n_errors'] for p in results[name]]),
                n_trials=np.array([p['n_trials'] for p in results[name]]),
                snr_db=float(snr_db),
                method=str(name),
                seed=int(seed),
            )

            reliable = "OK" if is_reliable(r) else "LOW"
            print(f"  [{name}] INR={inr_db:.1f}dB -> BLER={r['bler']:.2e} "
                  f"({r['n_errors']}/{r['n_trials']} errors) [{reliable}]")

    return results


def _save_method_results(
    pts: list[dict],
    method_name: str,
    inr_db: float,
    results_dir: Path,
    tag: str,
    seed: int,
) -> None:
    """Save method results to .npz file."""
    save_path = results_dir / f"{tag}_{method_name}_inr{inr_db:.0f}dB.npz"
    np.savez(
        save_path,
        snr_db=np.array([p['snr_db'] for p in pts]),
        bler=np.array([p['bler'] for p in pts]),
        ci_95=np.array([p['ci_95'] for p in pts]),
        n_errors=np.array([p['n_errors'] for p in pts]),
        n_trials=np.array([p['n_trials'] for p in pts]),
        inr_db=float(inr_db),
        method=str(method_name),
        seed=int(seed),
        timestamp=str(datetime.now().isoformat()),
    )


def _load_snr_sweep(npz_path: Path) -> list[dict]:
    """Load a saved SNR sweep .npz as a list of estimate_bler-style dicts."""
    d = np.load(npz_path)
    return [
        dict(snr_db=float(d['snr_db'][i]), bler=float(d['bler'][i]),
             ci_95=float(d['ci_95'][i]), n_errors=int(d['n_errors'][i]),
             n_trials=int(d['n_trials'][i]), inr_db=float(d['inr_db']))
        for i in range(len(d['snr_db']))
    ]


def _load_inr_sweep(npz_path: Path) -> list[dict]:
    """Load a saved INR sweep .npz as a list of estimate_bler-style dicts."""
    d = np.load(npz_path)
    return [
        dict(inr_db=float(d['inr_db'][i]), bler=float(d['bler'][i]),
             ci_95=float(d['ci_95'][i]), n_errors=int(d['n_errors'][i]),
             n_trials=int(d['n_trials'][i]), snr_db=float(d['snr_db']))
        for i in range(len(d['inr_db']))
    ]


# ─────────────────────────────────────────────
# Phase 1 main evaluation
# ─────────────────────────────────────────────
def run_phase1(
    n_trials: int = 10000,
    snr_range: Optional[np.ndarray] = None,
    inr_db: float = 5.0,
    seed: int = 42,
) -> None:
    """
    Run Phase 1 evaluation: B1, B2, B5 baselines.

    Args:
        n_trials: Monte Carlo trials per SNR point
        snr_range: SNR values to evaluate (default: 0-10 dB, step 1)
        inr_db: interference level in dB
        seed: random seed
    """
    if snr_range is None:
        snr_range = np.arange(0, 11, 1, dtype=float)

    print("=" * 60)
    print("Phase 1: Classical Baselines Evaluation")
    print(f"  INR = {inr_db} dB, n_trials = {n_trials}, seed = {seed}")
    print("=" * 60)

    trellis = load_nasa_k7()
    encode_fn = make_encoder(trellis)

    methods = {
        'B1_mismatched_viterbi': (encode_fn, make_decoder_b1(trellis)),
        'B2_oracle_viterbi': (encode_fn, make_decoder_b2(trellis)),
        'B5_interference_cancel': (encode_fn, make_decoder_b5(trellis)),
    }

    # Results directory
    phase1_dir = RESULTS_DIR / "phase1"
    fig_dir = phase1_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Run sweep
    results = sweep_snr(
        methods, snr_range, inr_db, n_trials,
        results_dir=phase1_dir, tag="bler", seed=seed,
    )

    # Add theoretical uncoded BPSK curve
    snr_lin = 10 ** (snr_range / 10)
    bpsk_ber = 0.5 * erfc(np.sqrt(snr_lin))
    results['Uncoded_BPSK_theory'] = [
        dict(snr_db=float(s), bler=float(b), ci_95=0.0,
             n_errors=0, n_trials=0, inr_db=inr_db)
        for s, b in zip(snr_range, bpsk_ber)
    ]

    # Plot
    plot_bler_vs_snr(
        results, inr_db=inr_db,
        save_path=fig_dir / "phase1_bler_vs_snr",
        title="Phase 1 Baselines",
    )

    # ── INR sweep at fixed SNR ──
    inr_range = np.arange(-5, 15.01, 2.5)
    snr_fixed = 5.0
    print("\n" + "=" * 60)
    print(f"INR Sweep: SNR = {snr_fixed} dB, INR = {inr_range[0]:.0f}–{inr_range[-1]:.0f} dB")
    print("=" * 60)

    inr_results = sweep_inr(
        methods, inr_range, snr_db=snr_fixed, n_trials=n_trials,
        results_dir=phase1_dir, tag="bler_inr", seed=seed,
    )

    plot_bler_vs_inr(
        inr_results, snr_db=snr_fixed,
        save_path=fig_dir / "phase1_bler_vs_inr",
    )

    # Compute cost table
    print("\n" + "=" * 60)
    print("Compute Cost Table")
    print("=" * 60)

    viterbi_flops = count_flops_viterbi(N_STATES, N_CODED)
    ic_viterbi_flops = count_flops_ic_viterbi(N_STATES, N_CODED)

    # Get BLER at 5 dB
    def get_bler_at_snr(pts, target_snr=5.0):
        for p in pts:
            if abs(p['snr_db'] - target_snr) < 0.1:
                return p['bler']
        return float('nan')

    cost_methods = {
        'B1_mismatched_viterbi': {
            'flops': viterbi_flops,
            'latency_ms': 0.0,
            'bler_at_5db': get_bler_at_snr(results['B1_mismatched_viterbi']),
        },
        'B2_oracle_viterbi': {
            'flops': viterbi_flops,
            'latency_ms': 0.0,
            'bler_at_5db': get_bler_at_snr(results['B2_oracle_viterbi']),
        },
        'B5_interference_cancel': {
            'flops': ic_viterbi_flops,
            'latency_ms': 0.0,
            'bler_at_5db': get_bler_at_snr(results['B5_interference_cancel']),
        },
    }

    # Measure latency
    rng = np.random.default_rng(0)
    test_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
    test_symbols = encode_fn(test_bits)
    test_rx = awgn_channel(test_symbols, 5.0, inr_db, 16.0, 0.0, rng)

    for name, (_, decode_fn) in methods.items():
        lat = measure_latency_ms(
            lambda rx: decode_fn(rx, 16.0, 0.0, 5.0, inr_db),
            test_rx, n_warmup=5, n_runs=20,
        )
        cost_methods[name]['latency_ms'] = lat['mean_ms']

    table = make_compute_table(cost_methods)
    print(table)

    # Budget checks
    print()
    assert_within_budget("B1_mismatched_viterbi", viterbi_flops)
    assert_within_budget("B2_oracle_viterbi", viterbi_flops)
    assert_within_budget("B5_interference_cancel", ic_viterbi_flops)

    # Save compute table
    with open(phase1_dir / "compute_table.md", 'w') as f:
        f.write(table)

    print(f"\nResults saved to: {phase1_dir}")
    print(f"Figures saved to: {fig_dir}")


# ─────────────────────────────────────────────
# Phase 2a: Neural decoder evaluation
# ─────────────────────────────────────────────
def run_phase2a(
    checkpoint_path: Optional[str] = None,
    n_trials: int = 100000,
    snr_range: Optional[np.ndarray] = None,
    inr_db: float = 5.0,
    seed: int = 42,
    device: str = "",
) -> None:
    """
    Run Phase 2a evaluation: N1 (BiGRU) vs B1, B2, B5 baselines.

    Args:
        checkpoint_path: path to trained BiGRU checkpoint
        n_trials: Monte Carlo trials per point
        snr_range: SNR values to evaluate
        inr_db: interference level in dB
        seed: random seed
        device: torch device (auto-detect if empty)
    """
    import torch
    # from neural_decoder import load_model as load_n1, make_decoder_n1  # N1 not trained yet
    from neural_bm import (
        load_model as load_n2_model, make_decoder_n2,
        build_branch_output_index,
    )
    from compute_cost import (
        count_flops_birnn_analytical, count_flops_neural,
    )

    if snr_range is None:
        snr_range = np.arange(0, 11, 1, dtype=float)

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Default checkpoint path
    # N1 checkpoint (unused — N1 not trained yet)
    # if checkpoint_path is None:
    #     checkpoint_path = str(
    #         Path(__file__).parent / "results" / "phase2a" / "checkpoints" / "best_gru.pt"
    #     )

    n2_checkpoint_path = checkpoint_path or str(
        Path(__file__).parent / "results" / "phase2b" / "checkpoints" / "best_model_seed32.pt"
    )

    # Extract seed tag from checkpoint filename (e.g. "best_model_seed32.pt" -> "_seed32")
    import re
    _seed_match = re.search(r'seed(\d+)', Path(n2_checkpoint_path).stem)
    _seed_tag = f"_seed{_seed_match.group(1)}" if _seed_match else ""

    print("=" * 60)
    print("Phase 2a/2b: Neural Decoder Evaluation")
    print(f"  INR = {inr_db} dB, n_trials = {n_trials}, device = {device}")
    print(f"  N2 Checkpoint: {n2_checkpoint_path}")
    print("=" * 60)

    # Load N2 model
    n2_model, _ = load_n2_model(n2_checkpoint_path)

    # Build methods
    trellis = load_nasa_k7()
    encode_fn = make_encoder(trellis)

    index_table = build_branch_output_index(trellis)
    methods = {
        'B1_mismatched_viterbi': (encode_fn, make_decoder_b1(trellis)),
        'B2_oracle_viterbi':     (encode_fn, make_decoder_b2(trellis)),
        'B5_interference_cancel':(encode_fn, make_decoder_b5(trellis)),
        'N2_neural_bm':          (encode_fn, make_decoder_n2(n2_model, device, trellis, index_table)),
        # 'N1_gru_e2e':          (encode_fn, make_decoder_n1(model, device=device)),  # not trained yet
    }

    # Results directories
    phase1_dir = RESULTS_DIR / "phase1"
    phase2b_dir = RESULTS_DIR / "phase2b"
    fig_dir = phase2b_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── SNR sweep — load B1/B2/B5 from disk, only run N2 ──
    print("\nLoading Phase 1 SNR sweep results from disk...")
    results = {
        'B1_mismatched_viterbi':  _load_snr_sweep(phase1_dir / f"bler_B1_mismatched_viterbi_inr{inr_db:.0f}dB.npz"),
        'B2_oracle_viterbi':      _load_snr_sweep(phase1_dir / f"bler_B2_oracle_viterbi_inr{inr_db:.0f}dB.npz"),
        'B5_interference_cancel': _load_snr_sweep(phase1_dir / f"bler_B5_interference_cancel_inr{inr_db:.0f}dB.npz"),
    }
    print("Running N2 SNR sweep...")
    n2_snr = sweep_snr(
        {'N2_neural_bm': methods['N2_neural_bm']},
        snr_range, inr_db, n_trials,
        results_dir=phase2b_dir, tag="bler", seed=seed,
    )
    results['N2_neural_bm'] = n2_snr['N2_neural_bm']

    # Add theory curve
    snr_lin = 10 ** (snr_range / 10)
    bpsk_ber = 0.5 * erfc(np.sqrt(snr_lin))
    results['Uncoded_BPSK_theory'] = [
        dict(snr_db=float(s), bler=float(b), ci_95=0.0,
             n_errors=0, n_trials=0, inr_db=inr_db)
        for s, b in zip(snr_range, bpsk_ber)
    ]

    plot_bler_vs_snr(
        results, inr_db=inr_db,
        save_path=fig_dir / f"phase2b_bler_vs_snr{_seed_tag}",
        title="Phase 2b: N2 vs Classical",
    )

    # ── INR sweep — load B1/B2/B5 from disk, only run N2 ──
    inr_range = np.arange(-5, 15.01, 2.5)
    snr_fixed = 5.0
    print(f"\nLoading Phase 1 INR sweep results from disk...")
    inr_results = {
        'B1_mismatched_viterbi':  _load_inr_sweep(phase1_dir / f"bler_inr_B1_mismatched_viterbi_snr{snr_fixed:.0f}dB_inr_sweep.npz"),
        'B2_oracle_viterbi':      _load_inr_sweep(phase1_dir / f"bler_inr_B2_oracle_viterbi_snr{snr_fixed:.0f}dB_inr_sweep.npz"),
        'B5_interference_cancel': _load_inr_sweep(phase1_dir / f"bler_inr_B5_interference_cancel_snr{snr_fixed:.0f}dB_inr_sweep.npz"),
    }
    print(f"Running N2 INR sweep: SNR = {snr_fixed} dB")
    n2_inr = sweep_inr(
        {'N2_neural_bm': methods['N2_neural_bm']},
        inr_range, snr_db=snr_fixed, n_trials=n_trials,
        results_dir=phase2b_dir, tag="bler_inr", seed=seed,
    )
    inr_results['N2_neural_bm'] = n2_inr['N2_neural_bm']

    plot_bler_vs_inr(
        inr_results, snr_db=snr_fixed,
        save_path=fig_dir / f"phase2b_bler_vs_inr{_seed_tag}",
    )

    # ── Compute cost table ──
    print("\nCompute Cost Table:")
    viterbi_flops = count_flops_viterbi(N_STATES, N_CODED)
    ic_viterbi_flops = count_flops_ic_viterbi(N_STATES, N_CODED)
    n2_flops = count_flops_birnn_analytical(
        hidden_size=16, input_dim=2, seq_len=262,
        cell_type="GRU", bidirectional=True,
    )
    # gru_flops = count_flops_birnn_analytical(  # N1 — not used yet
    #     hidden_size=24, input_dim=2, seq_len=262,
    #     cell_type="GRU", bidirectional=True,
    # )

    def get_bler_at_snr(pts, target_snr=5.0):
        for p in pts:
            if abs(p['snr_db'] - target_snr) < 0.1:
                return p['bler']
        return float('nan')

    cost_methods = {
        'B1_mismatched_viterbi': {
            'flops': viterbi_flops, 'latency_ms': 0.0,
            'bler_at_5db': get_bler_at_snr(results['B1_mismatched_viterbi']),
        },
        'B2_oracle_viterbi': {
            'flops': viterbi_flops, 'latency_ms': 0.0,
            'bler_at_5db': get_bler_at_snr(results['B2_oracle_viterbi']),
        },
        'B5_interference_cancel': {
            'flops': ic_viterbi_flops, 'latency_ms': 0.0,
            'bler_at_5db': get_bler_at_snr(results['B5_interference_cancel']),
        },
        'N2_neural_bm': {
            'flops': n2_flops, 'latency_ms': 0.0,
            'bler_at_5db': get_bler_at_snr(results['N2_neural_bm']),
        },
        # 'N1_gru_e2e': {  # not trained yet
        #     'flops': gru_flops, 'latency_ms': 0.0,
        #     'bler_at_5db': get_bler_at_snr(results['N1_gru_e2e']),
        # },
    }

    # Measure latency
    rng = np.random.default_rng(0)
    test_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
    test_symbols = encode_fn(test_bits)
    test_rx = awgn_channel(test_symbols, 5.0, inr_db, 16.0, 0.0, rng)

    for name, (_, decode_fn) in methods.items():
        lat = measure_latency_ms(
            lambda rx: decode_fn(rx, 16.0, 0.0, 5.0, inr_db),
            test_rx, n_warmup=5, n_runs=20,
        )
        cost_methods[name]['latency_ms'] = lat['mean_ms']

    table = make_compute_table(cost_methods)
    print(table)

    # Budget checks
    print()
    assert_within_budget("B1_mismatched_viterbi", viterbi_flops)
    assert_within_budget("B2_oracle_viterbi", viterbi_flops)
    assert_within_budget("B5_interference_cancel", ic_viterbi_flops)
    assert_within_budget("N2_neural_bm", n2_flops)
    # assert_within_budget("N1_gru_e2e", gru_flops)  # not trained yet

    # Save compute table
    with open(phase2a_dir / "compute_table_phase2a.md", 'w') as f:
        f.write(table)

    # dB gain
    from plot_utils import db_gain
    try:
        snr_n2 = np.array([p['snr_db'] for p in results['N2_neural_bm']])
        bler_n2 = np.array([p['bler'] for p in results['N2_neural_bm']])
        snr_b1 = np.array([p['snr_db'] for p in results['B1_mismatched_viterbi']])
        bler_b1 = np.array([p['bler'] for p in results['B1_mismatched_viterbi']])
        gain = db_gain(1e-3, snr_n2, bler_n2, snr_b1, bler_b1)
        print(f"\nN2 vs B1 gain at BLER=1e-3: {gain:.2f} dB")
    except Exception as e:
        print(f"\nCould not compute dB gain: {e}")
    # N1 gain — not computed yet (model not trained)
    # try:
    #     snr_n1 = np.array([p['snr_db'] for p in results['N1_gru_e2e']])
    #     bler_n1 = np.array([p['bler'] for p in results['N1_gru_e2e']])
    #     gain = db_gain(1e-3, snr_n1, bler_n1, snr_b1, bler_b1)
    #     print(f"\nN1 vs B1 gain at BLER=1e-3: {gain:.2f} dB")
    # except Exception as e:
    #     print(f"\nCould not compute dB gain: {e}")

    print(f"\nResults saved to: {phase2a_dir}")
    print(f"Figures saved to: {fig_dir}")


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BLER evaluation")
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Phase to run (1 or 2)')
    parser.add_argument('--n-trials', type=int, default=10000,
                        help='Monte Carlo trials per SNR point')
    parser.add_argument('--inr-db', type=float, default=5.0,
                        help='Interference-to-noise ratio (dB)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--snr-min', type=float, default=0.0)
    parser.add_argument('--snr-max', type=float, default=10.0)
    parser.add_argument('--snr-step', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained neural decoder checkpoint')
    parser.add_argument('--device', type=str, default='',
                        help='torch device (auto-detect if empty)')
    args = parser.parse_args()

    snr_range = np.arange(args.snr_min, args.snr_max + 0.01, args.snr_step)

    if args.phase == 1:
        run_phase1(
            n_trials=args.n_trials,
            snr_range=snr_range,
            inr_db=args.inr_db,
            seed=args.seed,
        )
    elif args.phase == 2:
        run_phase2a(
            checkpoint_path=args.checkpoint,
            n_trials=args.n_trials,
            snr_range=snr_range,
            inr_db=args.inr_db,
            seed=args.seed,
            device=args.device,
        )
