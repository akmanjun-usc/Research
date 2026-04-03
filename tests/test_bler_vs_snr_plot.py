"""
test_bler_vs_snr_plot.py — BLER vs SNR comparison with 5 curves

Plots:
  1. Mismatched Viterbi — AWGN only (no interference)
  2. Oracle Viterbi — AWGN only (no interference)
  3. Mismatched Viterbi — AWGN + sinusoidal interference (INR=5 dB)
  4. Oracle Viterbi — AWGN + sinusoidal interference (INR=5 dB)
  5. Theoretical uncoded BPSK BER: Q(sqrt(2·SNR_linear))

Part of: EE597 Search-Designed Trellis Codes Project
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.special import erfc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from channel import awgn_channel
from trellis import load_nasa_k7
from eval import estimate_bler, make_encoder, make_decoder_b1, make_decoder_b2
from plot_utils import set_paper_style


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SNR_RANGE = np.arange(0, 11, 1, dtype=float)
N_TRIALS = 5000
INR_DB = 5.0
INR_OFF = -200.0  # effectively zero interference
SEED = 42
SAVE_DIR = Path(__file__).resolve().parent.parent / "results" / "test"


# ─────────────────────────────────────────────
# Theoretical uncoded BPSK BER
# ─────────────────────────────────────────────
def theoretical_bpsk_ber(snr_db: np.ndarray) -> np.ndarray:
    """Q(sqrt(2·SNR_linear)) = 0.5·erfc(sqrt(SNR_linear))."""
    snr_lin = 10 ** (snr_db / 10)
    return 0.5 * erfc(np.sqrt(snr_lin))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("BLER vs SNR — 5-curve comparison")
    print(f"  SNR: {SNR_RANGE[0]:.0f}–{SNR_RANGE[-1]:.0f} dB, "
          f"n_trials={N_TRIALS}, INR={INR_DB} dB")
    print("=" * 60)

    trellis = load_nasa_k7()
    encode_fn = make_encoder(trellis)
    decode_b1 = make_decoder_b1(trellis)
    decode_b2 = make_decoder_b2(trellis)

    # Storage: {label: (snr_list, bler_list)}
    curves: dict[str, tuple[list[float], list[float]]] = {}

    # ── Curve 1: Mismatched Viterbi, AWGN only ──
    label = "Mismatch Viterbi (AWGN only)"
    print(f"\n[{label}]")
    snrs, blers = [], []
    for snr_db in SNR_RANGE:
        r = estimate_bler(encode_fn, decode_b1, awgn_channel,
                          snr_db=float(snr_db), inr_db=INR_OFF,
                          n_trials=N_TRIALS, seed=SEED)
        snrs.append(snr_db)
        blers.append(r['bler'])
        print(f"  SNR={snr_db:.0f} dB  BLER={r['bler']:.4e}  "
              f"({r['n_errors']}/{r['n_trials']})")
    curves[label] = (snrs, blers)

    # ── Curve 2: Oracle Viterbi, AWGN only ──
    label = "Oracle Viterbi (AWGN only)"
    print(f"\n[{label}]")
    snrs, blers = [], []
    for snr_db in SNR_RANGE:
        r = estimate_bler(encode_fn, decode_b2, awgn_channel,
                          snr_db=float(snr_db), inr_db=INR_OFF,
                          n_trials=N_TRIALS, seed=SEED)
        snrs.append(snr_db)
        blers.append(r['bler'])
        print(f"  SNR={snr_db:.0f} dB  BLER={r['bler']:.4e}  "
              f"({r['n_errors']}/{r['n_trials']})")
    curves[label] = (snrs, blers)

    # ── Curve 3: Mismatched Viterbi, AWGN + interference ──
    label = f"Mismatch Viterbi (INR={INR_DB:.0f} dB)"
    print(f"\n[{label}]")
    snrs, blers = [], []
    for snr_db in SNR_RANGE:
        r = estimate_bler(encode_fn, decode_b1, awgn_channel,
                          snr_db=float(snr_db), inr_db=INR_DB,
                          n_trials=N_TRIALS, seed=SEED)
        snrs.append(snr_db)
        blers.append(r['bler'])
        print(f"  SNR={snr_db:.0f} dB  BLER={r['bler']:.4e}  "
              f"({r['n_errors']}/{r['n_trials']})")
    curves[label] = (snrs, blers)

    # ── Curve 4: Oracle Viterbi, AWGN + interference ──
    label = f"Oracle Viterbi (INR={INR_DB:.0f} dB)"
    print(f"\n[{label}]")
    snrs, blers = [], []
    for snr_db in SNR_RANGE:
        r = estimate_bler(encode_fn, decode_b2, awgn_channel,
                          snr_db=float(snr_db), inr_db=INR_DB,
                          n_trials=N_TRIALS, seed=SEED)
        snrs.append(snr_db)
        blers.append(r['bler'])
        print(f"  SNR={snr_db:.0f} dB  BLER={r['bler']:.4e}  "
              f"({r['n_errors']}/{r['n_trials']})")
    curves[label] = (snrs, blers)

    # ── Curve 5: Theoretical uncoded BPSK ──
    theory_snr = np.linspace(0, 10, 100)
    theory_ber = theoretical_bpsk_ber(theory_snr)

    # ── Plot ──
    print("\nGenerating plot...")
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Plot style for each curve
    styles = {
        "Mismatch Viterbi (AWGN only)":
            dict(color='#2ca02c', marker='s', ls='-', lw=1.5),
        "Oracle Viterbi (AWGN only)":
            dict(color='#1f77b4', marker='D', ls='-', lw=1.5),
        f"Mismatch Viterbi (INR={INR_DB:.0f} dB)":
            dict(color='#d62728', marker='o', ls='--', lw=1.5),
        f"Oracle Viterbi (INR={INR_DB:.0f} dB)":
            dict(color='#ff7f0e', marker='^', ls='--', lw=1.5),
    }

    for label, (snrs, blers) in curves.items():
        # Filter zero-BLER points (can't plot on log scale)
        plot_snr = [s for s, b in zip(snrs, blers) if b > 0]
        plot_bler = [b for b in blers if b > 0]
        if not plot_bler:
            continue
        st = styles[label]
        ax.semilogy(plot_snr, plot_bler, label=label, **st)

    # Theoretical curve
    ax.semilogy(theory_snr, theory_ber, 'k:', lw=1.0,
                label='Uncoded BPSK (theory)')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Block Error Rate (BLER)')
    ax.set_title('BLER vs SNR — Viterbi Baselines')
    ax.set_xlim([0, 10])
    ax.set_ylim([1e-4, 1.0])
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_base = SAVE_DIR / "bler_vs_snr_comparison"
    fig.savefig(save_base.with_suffix('.png'), bbox_inches='tight', dpi=200)
    fig.savefig(save_base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved: {save_base.with_suffix('.png')}")
    print(f"Saved: {save_base.with_suffix('.pdf')}")

    # ── Sanity checks ──
    print("\n" + "=" * 60)
    print("Sanity checks")
    print("=" * 60)

    awgn_mm = dict(zip(
        curves["Mismatch Viterbi (AWGN only)"][0],
        curves["Mismatch Viterbi (AWGN only)"][1],
    ))
    awgn_or = dict(zip(
        curves["Oracle Viterbi (AWGN only)"][0],
        curves["Oracle Viterbi (AWGN only)"][1],
    ))
    intf_mm = dict(zip(
        curves[f"Mismatch Viterbi (INR={INR_DB:.0f} dB)"][0],
        curves[f"Mismatch Viterbi (INR={INR_DB:.0f} dB)"][1],
    ))
    intf_or = dict(zip(
        curves[f"Oracle Viterbi (INR={INR_DB:.0f} dB)"][0],
        curves[f"Oracle Viterbi (INR={INR_DB:.0f} dB)"][1],
    ))

    # Check 1: Oracle ≈ Mismatched in AWGN-only (no interference to mismatch on)
    max_diff = max(abs(awgn_mm[s] - awgn_or[s]) for s in SNR_RANGE)
    status = "PASS" if max_diff < 0.05 else "WARN"
    print(f"  [{status}] AWGN-only: Oracle ≈ Mismatched (max diff = {max_diff:.4f})")

    # Check 2: Oracle w/ interference ≤ Mismatched w/ interference
    violations = sum(1 for s in SNR_RANGE if intf_or[s] > intf_mm[s] + 0.02)
    status = "PASS" if violations == 0 else "WARN"
    print(f"  [{status}] Interference: Oracle ≤ Mismatched "
          f"({violations} violations out of {len(SNR_RANGE)} points)")

    # Check 3: Interference hurts mismatched decoder
    worse_count = sum(1 for s in SNR_RANGE
                      if intf_mm[s] > awgn_mm[s] and awgn_mm[s] > 0)
    status = "PASS" if worse_count > 0 else "WARN"
    print(f"  [{status}] Interference degrades mismatched Viterbi "
          f"({worse_count}/{len(SNR_RANGE)} points worse)")

    print("\nDone.")


if __name__ == "__main__":
    main()
