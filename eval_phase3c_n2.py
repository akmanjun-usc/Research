"""
eval_phase3c_n2.py — BLER sweep for the SA-found trellis + N2 neural BM decoder (Phase 3c)

Loads best_trellis_seed0.npz from results/phase3c_n2/, runs SNR and INR sweeps
with the N2 decoder, and compares against B1, B2, B5, N2+NASA K=7, and S1-oracle.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

RESULTS_DIR = Path("results")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3c N2 eval")
    parser.add_argument("--n-trials", type=int, default=10_000)
    parser.add_argument("--inr-db", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--genome", type=str, default="results/phase3c_n2/best_trellis_seed0.npz")
    parser.add_argument("--checkpoint", type=str, default="results/phase2b/checkpoints/best_model_seed32.pt")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    inr_db = args.inr_db
    snr_range = np.arange(0, 11, 1, dtype=float)
    snr_fixed = 5.0
    inr_range = np.arange(-5, 15.01, 2.5)

    from eval import (
        make_encoder, sweep_snr, sweep_inr,
        _load_snr_sweep, _load_inr_sweep, _get_bler_at_snr,
        make_decoder_b1, make_decoder_b2, make_decoder_b5,
    )
    from trellis_genome import genome_to_trellis
    from trellis import load_nasa_k7
    from neural_bm import load_model as load_n2_model, make_decoder_n2, build_branch_output_index
    from plot_utils import plot_bler_vs_snr, plot_bler_vs_inr, db_gain

    genome_path = Path(args.genome)
    n2_checkpoint = Path(args.checkpoint)

    print("=" * 60)
    print("Phase 3c: SA-found Trellis + N2 Neural BM Evaluation")
    print(f"  Genome:     {genome_path}")
    print(f"  Checkpoint: {n2_checkpoint}")
    print(f"  INR = {inr_db} dB,  n_trials = {args.n_trials},  seed = {args.seed},  device = {dev}")
    print("=" * 60)

    bt = np.load(genome_path)
    genome = {"next_state": bt["next_state"], "output_pair": bt["output_pair"], "n_states": 64}
    trellis = genome_to_trellis(genome)
    print(f"  SA proxy fitness (BM alignment energy): {float(bt['fitness']):.4e}\n")

    n2_model, _ = load_n2_model(n2_checkpoint)
    index_table = build_branch_output_index(trellis)
    encode_fn   = make_encoder(trellis)
    decode_fn   = make_decoder_n2(n2_model, dev, trellis, index_table)

    phase3c_dir = RESULTS_DIR / "phase3c_n2"
    phase3b_dir = RESULTS_DIR / "phase3b"
    phase3a_dir = RESULTS_DIR / "phase3a"
    phase2b_dir = RESULTS_DIR / "phase2b"
    phase1_dir  = RESULTS_DIR / "phase1"
    fig_dir     = phase3c_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── SNR sweep ─────────────────────────────────────────────────────────────
    print("Loading Phase 1 SNR sweep results from disk...")
    snr_results = {
        'B1_mismatched_viterbi':  _load_snr_sweep(phase1_dir / f"bler_B1_mismatched_viterbi_inr{inr_db:.0f}dB.npz"),
        'B2_oracle_viterbi':      _load_snr_sweep(phase1_dir / f"bler_B2_oracle_viterbi_inr{inr_db:.0f}dB.npz"),
        'B5_interference_cancel': _load_snr_sweep(phase1_dir / f"bler_B5_interference_cancel_inr{inr_db:.0f}dB.npz"),
    }
    p2b_snr = phase2b_dir / f"bler_N2_neural_bm_inr{inr_db:.0f}dB_seed32.npz"
    if p2b_snr.exists():
        print("Loading Phase 2b SNR sweep results from disk...")
        snr_results['N2_nasa_neural_bm'] = _load_snr_sweep(p2b_snr)
    p3a_snr = phase3a_dir / f"bler_S1_oracle_viterbi_inr{inr_db:.0f}dB.npz"
    if p3a_snr.exists():
        print("Loading Phase 3a SNR sweep results from disk...")
        snr_results['S1_oracle_viterbi'] = _load_snr_sweep(p3a_snr)
    p3b_snr = phase3b_dir / f"bler_S1_neural_bm_inr{inr_db:.0f}dB.npz"
    if p3b_snr.exists():
        print("Loading Phase 3b SNR sweep results from disk...")
        snr_results['S1_ea_neural_bm'] = _load_snr_sweep(p3b_snr)

    print("Running SA-found trellis N2 SNR sweep...")
    sa_snr = sweep_snr(
        {'S1_sa_neural_bm': (encode_fn, decode_fn)},
        snr_range, inr_db, args.n_trials,
        results_dir=phase3c_dir, tag="bler", seed=args.seed,
    )
    snr_results['S1_sa_neural_bm'] = sa_snr['S1_sa_neural_bm']

    plot_bler_vs_snr(
        snr_results, inr_db=inr_db,
        save_path=fig_dir / "phase3c_bler_vs_snr",
        title="Phase 3c: SA Trellis + N2 vs Baselines",
    )

    # ── INR sweep ─────────────────────────────────────────────────────────────
    print(f"\nLoading Phase 1 INR sweep results from disk...")
    inr_results = {
        'B1_mismatched_viterbi':  _load_inr_sweep(phase1_dir / f"bler_inr_B1_mismatched_viterbi_snr{snr_fixed:.0f}dB_inr_sweep.npz"),
        'B2_oracle_viterbi':      _load_inr_sweep(phase1_dir / f"bler_inr_B2_oracle_viterbi_snr{snr_fixed:.0f}dB_inr_sweep.npz"),
        'B5_interference_cancel': _load_inr_sweep(phase1_dir / f"bler_inr_B5_interference_cancel_snr{snr_fixed:.0f}dB_inr_sweep.npz"),
    }
    p2b_inr = phase2b_dir / f"bler_inr_N2_neural_bm_snr{snr_fixed:.0f}dB_inr_sweep_seed32.npz"
    if p2b_inr.exists():
        print("Loading Phase 2b INR sweep results from disk...")
        inr_results['N2_nasa_neural_bm'] = _load_inr_sweep(p2b_inr)
    p3a_inr = phase3a_dir / f"bler_inr_S1_oracle_viterbi_snr{snr_fixed:.0f}dB_inr_sweep.npz"
    if p3a_inr.exists():
        inr_results['S1_oracle_viterbi'] = _load_inr_sweep(p3a_inr)
    p3b_inr = phase3b_dir / f"bler_inr_S1_neural_bm_snr{snr_fixed:.0f}dB_inr_sweep.npz"
    if p3b_inr.exists():
        inr_results['S1_ea_neural_bm'] = _load_inr_sweep(p3b_inr)

    print(f"Running SA-found trellis N2 INR sweep: SNR = {snr_fixed} dB")
    sa_inr = sweep_inr(
        {'S1_sa_neural_bm': (encode_fn, decode_fn)},
        inr_range, snr_db=snr_fixed, n_trials=args.n_trials,
        results_dir=phase3c_dir, tag="bler_inr", seed=args.seed,
    )
    inr_results['S1_sa_neural_bm'] = sa_inr['S1_sa_neural_bm']

    plot_bler_vs_inr(
        inr_results, snr_db=snr_fixed,
        save_path=fig_dir / "phase3c_bler_vs_inr",
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n--- SNR sweep summary (INR = {:.0f} dB) ---".format(inr_db))
    print(f"  {'Method':<30} {'BLER @ 5 dB':>12}")
    for name, pts in snr_results.items():
        print(f"  {name:<30} {_get_bler_at_snr(pts):>12.3e}")

    if 'N2_nasa_neural_bm' in snr_results:
        try:
            snr_sa   = np.array([p['snr_db'] for p in snr_results['S1_sa_neural_bm']])
            bler_sa  = np.array([p['bler']   for p in snr_results['S1_sa_neural_bm']])
            snr_n2   = np.array([p['snr_db'] for p in snr_results['N2_nasa_neural_bm']])
            bler_n2  = np.array([p['bler']   for p in snr_results['N2_nasa_neural_bm']])
            gain = db_gain(1e-3, snr_sa, bler_sa, snr_n2, bler_n2)
            print(f"\n  SA-trellis N2 vs N2+NASA gain at BLER=1e-3: {gain:+.2f} dB")
        except Exception as e:
            print(f"\n  Could not compute dB gain vs N2+NASA: {e}")

    print(f"\nFigures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
