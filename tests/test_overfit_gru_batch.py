"""
test_overfit_gru_batch.py — Overfit a single small batch with the BiGRU decoder.

Sanity-check: if the model can memorise a tiny fixed batch, the architecture
and training loop are working correctly.  Loss should approach ~0 and bit
accuracy should reach ~100 %.

Usage (run from the project root):
    python tests/test_overfit_gru_batch.py

No pytest runner needed — just plain Python.
"""

import sys
import os

# Allow imports from the project root regardless of where the script is called from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from neural_decoder import (
    BiRNNDecoder,
    generate_training_batch,
)
from trellis import load_nasa_k7
from channel import K_INFO


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 4        # Keep tiny so ~6.6K-param model can easily memorise
N_EPOCHS     = 500      # Gradient steps on the same fixed batch
LR           = 5e-3     # Learning rate
PRINT_EVERY  = 50       # Print loss every N epochs
SEED         = 42

# Fixed channel conditions: high SNR, low interference → clean signal
SNR_DB_FIXED = 10.0
INR_DB_FIXED  = 0.0
PERIOD_FIXED  = 16

# Thresholds for the final pass/fail summary
LOSS_THRESHOLD     = 0.05   # Final loss must be below this
ACCURACY_THRESHOLD = 0.95   # Bit accuracy must be above this


# ─────────────────────────────────────────────────────────────────────────────
# Helper: coloured terminal output (optional, degrades gracefully)
# ─────────────────────────────────────────────────────────────────────────────
def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"

def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(_bold(f"\n{'='*60}"))
    print(_bold(" BiGRU Decoder — Single-Batch Overfit Test"))
    print(_bold(f"{'='*60}"))
    print(f"  Device      : {device}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Epochs      : {N_EPOCHS}")
    print(f"  LR          : {LR}")
    print(f"  SNR (fixed) : {SNR_DB_FIXED} dB   |  INR: {INR_DB_FIXED} dB")
    print(f"  Period      : {PERIOD_FIXED}")
    print()

    trellis = load_nasa_k7()

    # ── Fix a single small batch (X_small, y_small) ─────────────────────────
    rng = np.random.default_rng(SEED)
    X_small, y_small = generate_training_batch(
        trellis=trellis,
        batch_size=BATCH_SIZE,
        snr_db=SNR_DB_FIXED,
        inr_db=INR_DB_FIXED,
        period_range=(PERIOD_FIXED, PERIOD_FIXED),
        rng=rng,
        device=device,
    )
    print(f"\n  Batch shapes : X_small={tuple(X_small.shape)}  "
          f"y_small={tuple(y_small.shape)}")
    print(f"  (X_small and y_small are FIXED throughout all {N_EPOCHS} epochs)\n")

    # ── Instantiate model ────────────────────────────────────────────────────
    model = BiRNNDecoder(
        hidden_size=32,
        input_dim=1,
        cell_type="GRU",
        bidirectional=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model        : BiGRU  h=32/dir  bidirectional=True")
    print(f"  Parameters   : {n_params:,}")
    print()

    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # ── Overfit loop — exact pattern requested ───────────────────────────────
    print(f"{'─'*50}")
    print(f"{'Epoch':>8}  {'Loss':>12}")
    print(f"{'─'*50}")

    model.train()
    loss_history = []

    for epoch in range(N_EPOCHS):
        output = model(X_small)
        loss   = criterion(output, y_small)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if epoch % PRINT_EVERY == 0:
            print(f"  Epoch {epoch:>4},  Loss: {loss_val:.6f}")

    print(f"{'─'*50}\n")

    # ── Final evaluation ─────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits   = model(X_small)
        probs    = torch.sigmoid(logits)
        preds    = (probs > 0.5).float()
        accuracy = (preds == y_small).float().mean().item()

    initial_loss = loss_history[0]
    final_loss   = loss_history[-1]
    loss_drop_pct = 100.0 * (initial_loss - final_loss) / (initial_loss + 1e-12)

    print(_bold("Results"))
    print(f"  Initial loss : {initial_loss:.6f}")
    print(f"  Final loss   : {final_loss:.6f}   (↓ {loss_drop_pct:.1f} %)")
    print(f"  Bit accuracy : {accuracy*100:.2f} %   "
          f"({int(accuracy * BATCH_SIZE * K_INFO)}/{BATCH_SIZE * K_INFO} bits correct)")
    print()

    # ── Pass / Fail summary ──────────────────────────────────────────────────
    passed = True
    checks = [
        ("Final loss < threshold",
         final_loss < LOSS_THRESHOLD,
         f"{final_loss:.6f} < {LOSS_THRESHOLD}"),
        ("Loss dropped ≥ 50 %",
         final_loss < initial_loss * 0.5,
         f"{initial_loss:.4f} → {final_loss:.4f}"),
        (f"Bit accuracy ≥ {ACCURACY_THRESHOLD*100:.0f} %",
         accuracy >= ACCURACY_THRESHOLD,
         f"{accuracy*100:.2f} %"),
    ]

    print("Checks:")
    for name, ok, detail in checks:
        icon = _green("✓ PASS") if ok else _red("✗ FAIL")
        print(f"  [{icon}]  {name:40s}  ({detail})")
        if not ok:
            passed = False

    print()
    if passed:
        print(_green(_bold("  ✓  OVERFIT TEST PASSED  — model can memorise a single batch.")))
    else:
        print(_red(_bold("  ✗  OVERFIT TEST FAILED  — check architecture / training loop.")))
    print()


if __name__ == "__main__":
    main()
