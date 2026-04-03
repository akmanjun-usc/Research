"""
plot_utils.py — Paper-quality figure helpers

Part of: EE597 Search-Designed Trellis Codes Project
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster compatibility
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# Paul Tol Bright palette — colorblind-safe
# ─────────────────────────────────────────────
TOL = {
    'blue':   '#4477AA',
    'red':    '#EE6677',
    'green':  '#228833',
    'yellow': '#CCBB44',
    'cyan':   '#66CCEE',
    'purple': '#AA3377',
    'grey':   '#BBBBBB',
}


# ─────────────────────────────────────────────
# IEEE-style rcParams — use via plt.rc_context()
# ─────────────────────────────────────────────
IEEE_RC = {
    'font.family':           'serif',
    'font.serif':            ['Times New Roman'],
    'font.size':             10,
    'axes.labelsize':        11,
    'axes.titlesize':        11,
    'xtick.labelsize':       9,
    'ytick.labelsize':       9,
    'legend.fontsize':       8.5,
    'legend.framealpha':     0.85,
    'legend.edgecolor':      '0.8',
    'legend.fancybox':       False,
    'figure.dpi':            150,
    'savefig.dpi':           300,
    'savefig.bbox':          'tight',
    'lines.linewidth':       1.5,
    'lines.markersize':      5,
    'axes.spines.top':       False,
    'axes.spines.right':     False,
    'grid.linewidth':        0.5,
    'grid.alpha':            0.7,
}


# ─────────────────────────────────────────────
# Method style map — consistent across ALL figures
# ─────────────────────────────────────────────
METHOD_STYLE = {
    'B1_mismatched_viterbi':  {'color': TOL['red'],    'marker': 'o',  'ls': '--',  'label': 'Mismatch Viterbi (B1)'},
    'B2_oracle_viterbi':      {'color': TOL['green'],  'marker': 's',  'ls': '--',  'label': 'Oracle Viterbi (B2)'},
    'B5_interference_cancel': {'color': TOL['yellow'], 'marker': '^',  'ls': '--',  'label': 'IC + Viterbi (B5)'},
    'N1_gru_e2e':             {'color': TOL['blue'],   'marker': 'D',  'ls': '-',   'label': 'GRU End-to-End (N1)'},
    'N2_neural_bm':           {'color': TOL['purple'], 'marker': 'v',  'ls': '-',   'label': 'Neural BM (N2)'},
    'S1_searched_gru':        {'color': TOL['cyan'],   'marker': '*',  'ls': '-',   'label': 'Searched + GRU (S1)', 'lw': 2.5},
    'B3_random_trellis':      {'color': TOL['grey'],   'marker': 'x',  'ls': ':',   'label': 'Random Trellis (B3)'},
    'Uncoded_BPSK_theory':    {'color': 'black',       'marker': '',   'ls': ':',   'label': 'Uncoded BPSK (theory)', 'lw': 1.0},
}


# ─────────────────────────────────────────────
# Legacy style setup (kept for backward compat)
# ─────────────────────────────────────────────
def set_paper_style() -> None:
    """IEEE-style plot configuration. Prefer plt.rc_context(IEEE_RC) instead."""
    sns.set_theme(style="ticks", font_scale=1.0)
    matplotlib.rcParams.update(IEEE_RC)


# ─────────────────────────────────────────────
# BLER vs SNR plot
# ─────────────────────────────────────────────
def plot_bler_vs_snr(
    results: dict[str, list[dict]],
    inr_db: float,
    save_path: Path,
    title: str = "",
    show_ci: bool = True,
) -> None:
    """
    Standard BLER vs SNR curve for all methods.

    Args:
        results: method_name -> list of estimate_bler output dicts
        inr_db: interference-to-noise ratio used
        save_path: path to save figure (both .pdf and .png)
        title: optional title prefix
        show_ci: whether to show 95% CI bands
    """
    sns.set_theme(style="ticks", font_scale=1.0)

    with plt.rc_context(IEEE_RC):
        fig, ax = plt.subplots(figsize=(7, 5.5))

        for method_name, pts in results.items():
            style = METHOD_STYLE.get(method_name, {})
            snr = [p['snr_db'] for p in pts]
            bler = [p['bler'] for p in pts]
            ci = [p['ci_95'] for p in pts]

            # Filter zero-BLER points (can't plot on log scale)
            plot_snr, plot_bler, plot_ci = [], [], []
            for s, b, c in zip(snr, bler, ci):
                if b > 0:
                    plot_snr.append(s)
                    plot_bler.append(b)
                    plot_ci.append(c)

            if not plot_bler:
                continue

            ax.semilogy(plot_snr, plot_bler,
                        color=style.get('color', 'black'),
                        marker=style.get('marker', 'o'),
                        ls=style.get('ls', '-'),
                        lw=style.get('lw', 1.5),
                        label=style.get('label', method_name))

            if show_ci and plot_ci:
                bler_arr = np.array(plot_bler)
                ci_arr = np.array(plot_ci)
                ax.fill_between(plot_snr,
                                np.clip(bler_arr - ci_arr, 1e-6, 1),
                                np.clip(bler_arr + ci_arr, 1e-6, 1),
                                alpha=0.15,
                                color=style.get('color', 'black'))

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Block Error Rate (BLER)')
        ax.set_ylim([1e-4, 1.0])
        ax.set_xlim([0, 10])
        ax.legend(loc='upper right')
        title_str = f'{title} (INR = {inr_db:.0f} dB)' if title else f'BLER vs SNR (INR = {inr_db:.0f} dB)'
        ax.set_title(title_str)
        ax.grid(True)
        sns.despine(ax=ax)

        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        fig.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path.with_suffix('.pdf')}")


# ─────────────────────────────────────────────
# BLER vs INR plot
# ─────────────────────────────────────────────
def plot_bler_vs_inr(
    results: dict[str, list[dict]],
    snr_db: float,
    save_path: Path,
) -> None:
    """Shows graceful degradation as interference strengthens."""
    sns.set_theme(style="ticks", font_scale=1.0)

    with plt.rc_context(IEEE_RC):
        fig, ax = plt.subplots(figsize=(7, 5.5))

        for method_name, pts in results.items():
            style = METHOD_STYLE.get(method_name, {})
            inr = [p['inr_db'] for p in pts]
            bler = [p['bler'] for p in pts]

            plot_inr, plot_bler = [], []
            for i, b in zip(inr, bler):
                if b > 0:
                    plot_inr.append(i)
                    plot_bler.append(b)

            if not plot_bler:
                continue

            ax.semilogy(plot_inr, plot_bler,
                        color=style.get('color', 'black'),
                        marker=style.get('marker', 'o'),
                        ls=style.get('ls', '-'),
                        lw=style.get('lw', 1.5),
                        label=style.get('label', method_name))

        ax.set_xlabel('INR (dB)')
        ax.set_ylabel('Block Error Rate (BLER)')
        ax.set_title(f'Robustness to interference (SNR = {snr_db:.0f} dB)')
        ax.legend(loc='lower right')
        ax.grid(True)
        sns.despine(ax=ax)

        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        fig.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path.with_suffix('.pdf')}")


# ─────────────────────────────────────────────
# dB gain calculation
# ─────────────────────────────────────────────
def db_gain(
    bler_target: float,
    snr_method: np.ndarray,
    bler_method: np.ndarray,
    snr_baseline: np.ndarray,
    bler_baseline: np.ndarray,
) -> float:
    """SNR difference at a fixed BLER target. Positive = improvement."""
    snr_m = np.interp(bler_target, bler_method[::-1], snr_method[::-1])
    snr_b = np.interp(bler_target, bler_baseline[::-1], snr_baseline[::-1])
    return snr_b - snr_m


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running smoke test for plot_utils...")

    # Create dummy results
    snr_range = np.arange(0, 11, 2)
    dummy_results = {
        'B1_mismatched_viterbi': [
            {'snr_db': s, 'bler': 0.5 * 10**(-s/10), 'ci_95': 0.01, 'n_errors': 100, 'n_trials': 1000, 'inr_db': 5.0}
            for s in snr_range
        ],
        'B2_oracle_viterbi': [
            {'snr_db': s, 'bler': 0.3 * 10**(-s/8), 'ci_95': 0.01, 'n_errors': 100, 'n_trials': 1000, 'inr_db': 5.0}
            for s in snr_range
        ],
    }

    save_dir = Path(__file__).parent / "results" / "test"
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_bler_vs_snr(dummy_results, inr_db=5.0,
                     save_path=save_dir / "test_bler_vs_snr")

    print("PASS")
