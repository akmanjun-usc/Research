"""
test_neural_overfit.py — Canonical pytest overfit sanity checks for N1.

This module is the automated regression test for verifying that the BiGRU
decoder can memorize a single fixed batch generated from the current
neural_decoder.py API.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

from neural_decoder import BiRNNDecoder, generate_training_batch
from trellis import load_nasa_k7


INPUT_DIM = 1
BATCH_SIZE = 4
N_STEPS = 500
LR = 5e-3
SEED = 42
SNR_DB_FIXED = 10.0
INR_DB_FIXED = 0.0
PERIOD_FIXED = (16, 16)


@pytest.fixture(scope="module")
def trellis():
    return load_nasa_k7()


@pytest.fixture(scope="module")
def fixed_batch(trellis):
    """One fixed batch at high SNR and low interference on CPU."""
    device = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    rx, bits = generate_training_batch(
        trellis=trellis,
        batch_size=BATCH_SIZE,
        snr_db=SNR_DB_FIXED,
        inr_db=INR_DB_FIXED,
        period_range=PERIOD_FIXED,
        rng=rng,
        device=device,
    )
    return rx, bits


def test_overfit_single_batch(fixed_batch):
    """Train on one fixed batch; loss must drop and accuracy must become high."""
    rx, bits = fixed_batch

    model = BiRNNDecoder(
        hidden_size=32,
        input_dim=INPUT_DIM,
        cell_type="GRU",
        bidirectional=True,
    )
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    initial_loss = None
    final_loss = None

    model.train()
    for step in range(N_STEPS):
        logits = model(rx)
        loss = criterion(logits, bits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == N_STEPS - 1:
            final_loss = loss.item()

    print(f"Overfit test: initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}")

    assert final_loss < 0.4, (
        f"Final loss {final_loss:.4f} > 0.4 — model failed to overfit"
    )
    assert final_loss < initial_loss * 0.5, (
        f"Loss didn't drop enough: {initial_loss:.4f} -> {final_loss:.4f}"
    )

    model.eval()
    with torch.no_grad():
        logits = model(rx)
        preds = (torch.sigmoid(logits) > 0.5).float()
        accuracy = (preds == bits).float().mean().item()

    print(f"Overfit test: bit accuracy = {accuracy:.4f}")
    assert accuracy > 0.85, f"Bit accuracy {accuracy:.4f} < 0.85"


def test_loss_decreases(fixed_batch):
    """Training loss should decrease materially on the same fixed batch."""
    rx, bits = fixed_batch

    model = BiRNNDecoder(
        hidden_size=32,
        input_dim=INPUT_DIM,
        cell_type="GRU",
        bidirectional=True,
    )
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    model.train()
    for _ in range(200):
        logits = model(rx)
        loss = criterion(logits, bits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first_avg = np.mean(losses[:20])
    last_avg = np.mean(losses[-20:])

    assert last_avg < first_avg * 0.8, (
        f"Loss didn't decrease: first_20={first_avg:.4f}, last_20={last_avg:.4f}"
    )
