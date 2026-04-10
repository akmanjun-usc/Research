"""
plot_training_history.py — Plot N1 or N2 training history from saved .npz files.

Usage:
    python plot_training_history.py --model N2
    python plot_training_history.py --model N2 --history results/phase2b/logs/history_seed42.npz
    python plot_training_history.py --model N1 --history results/phase2a/logs/history_seed42.npz
    python plot_training_history.py --model N1 --out results/phase2a/figures/training_history_seed42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import TOL, IEEE_RC


def plot_training_history_n2(history_path: Path, out_path: Path) -> None:
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


def plot_training_history_n1(history_path: Path, out_path: Path) -> None:
    h = np.load(history_path)

    train_loss = h['train_loss']
    train_bit_acc = h['train_bit_acc']
    val_bler = h['val_bler']
    val_epoch = h['val_epoch']

    epochs = np.arange(1, len(train_loss) + 1)

    sns.set_theme(style="ticks", font_scale=1.0)

    with plt.rc_context(IEEE_RC):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        ax.plot(epochs, train_loss, color=TOL['blue'], lw=1.5, label='train loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BCE Loss')
        ax.set_title('N1 Train Loss')
        ax.grid(True)
        ax.legend()

        ax = axes[1]
        ax.plot(epochs, train_bit_acc, color=TOL['green'], lw=1.5, label='train bit acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Bit Accuracy')
        ax.set_title('N1 Train Bit Accuracy')
        ax.set_ylim(0.0, 1.0)
        ax.grid(True)
        ax.legend()

        ax = axes[2]
        ax.semilogy(
            val_epoch,
            val_bler,
            color=TOL['purple'],
            marker='o',
            markersize=4,
            lw=1.5,
            label='val BLER',
        )
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Block Error Rate (BLER)')
        ax.set_title('Validation BLER (SNR=5 dB, INR=5 dB)')
        ax.grid(True)
        ax.legend()

        fig.suptitle('N1 Training History', fontsize=11)
        fig.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path.with_suffix('.pdf'))
        fig.savefig(out_path.with_suffix('.png'))
        print(f"Saved: {out_path.with_suffix('.pdf')}")
        print(f"Saved: {out_path.with_suffix('.png')}")
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot N1 or N2 training history')
    parser.add_argument(
        '--model',
        type=str,
        choices=['N1', 'N2'],
        required=True,
        help='Training history format to plot',
    )
    parser.add_argument(
        '--history',
        type=Path,
        default=None,
        help='Path to history .npz file',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=None,
        help='Output path (without extension)',
    )
    args = parser.parse_args()

    if args.model == 'N1':
        history_path = args.history or Path('results/phase2a/logs/history_seed42.npz')
        out_path = args.out or Path('results/phase2a/figures/training_history_seed42')
        plot_training_history_n1(history_path, out_path)
    else:
        history_path = args.history or Path('results/phase2b/logs/history_seed42.npz')
        out_path = args.out or Path('results/phase2b/figures/training_history_seed42')
        plot_training_history_n2(history_path, out_path)
