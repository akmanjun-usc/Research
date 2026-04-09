"""
plot_training_history.py — Plot N2 training history from saved .npz file.

Usage:
    python plot_training_history.py
    python plot_training_history.py --history results/phase2b/logs/history_seed42.npz
    python plot_training_history.py --out results/phase2b/figures/training_history
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import TOL, IEEE_RC


def plot_training_history(history_path: Path, out_path: Path) -> None:
    h = np.load(history_path)

    train_mse    = h['train_mse']          # (num_epochs,)
    mse_h0       = h['train_mse_h0']       # (num_epochs,)
    mse_h1       = h['train_mse_h1']
    mse_h2       = h['train_mse_h2']
    mse_h3       = h['train_mse_h3']
    val_bler     = h['val_bler']            # (num_val_steps,)
    val_epoch    = h['val_epoch']           # (num_val_steps,)

    epochs = np.arange(1, len(train_mse) + 1)

    sns.set_theme(style="ticks", font_scale=1.0)

    with plt.rc_context(IEEE_RC):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # ── Panel 1: Overall train MSE ────────────────────────────────────
        ax = axes[0]
        ax.plot(epochs, train_mse, color=TOL['blue'], lw=1.5, label='train MSE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.set_title('Overall Train MSE')
        ax.grid(True)
        ax.legend()

        # ── Panel 2: Per-hypothesis train MSE ─────────────────────────────
        ax = axes[1]
        hyp_colors = [TOL['blue'], TOL['red'], TOL['green'], TOL['yellow']]
        hyp_labels = [
            r'h0: $(-1,-1)$',
            r'h1: $(-1,+1)$',
            r'h2: $(+1,-1)$',
            r'h3: $(+1,+1)$',
        ]
        for mse_hyp, color, label in zip(
            [mse_h0, mse_h1, mse_h2, mse_h3], hyp_colors, hyp_labels
        ):
            ax.plot(epochs, mse_hyp, color=color, lw=1.5, label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.set_title('Per-Hypothesis Train MSE')
        ax.grid(True)
        ax.legend()

        # ── Panel 3: Validation BLER ──────────────────────────────────────
        ax = axes[2]
        ax.semilogy(val_epoch, val_bler,
                    color=TOL['purple'], marker='o', markersize=4,
                    lw=1.5, label='val BLER')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Block Error Rate (BLER)')
        ax.set_title('Validation BLER (SNR=5 dB, INR=5 dB)')
        ax.grid(True)
        ax.legend()

        fig.suptitle('N2 Training History', fontsize=11)
        fig.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path.with_suffix('.pdf'))
        fig.savefig(out_path.with_suffix('.png'))
        print(f"Saved: {out_path.with_suffix('.pdf')}")
        print(f"Saved: {out_path.with_suffix('.png')}")
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot N2 training history')
    parser.add_argument(
        '--history',
        type=Path,
        default=Path('results/phase2b/logs/history_seed42.npz'),
        help='Path to history .npz file',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('results/phase2b/figures/training_history'),
        help='Output path (without extension)',
    )
    args = parser.parse_args()
    plot_training_history(args.history, args.out)
