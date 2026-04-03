"""
test_neural_overfit.py — Verify BiGRU can overfit a single batch

Sanity check: if the model cannot memorize a small batch, the architecture
or training loop has a bug.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from neural_decoder import (
    BiRNNDecoder, build_codeword_pool, generate_training_batch, K_INFO,
)
from trellis import load_nasa_k7


N_SYM = 524
INPUT_DIM = 1
BATCH_SIZE = 4  # Small so model (6.6K params) can memorize (4*256=1024 outputs)
N_STEPS = 500
LR = 5e-3


@pytest.fixture(scope="module")
def trellis():
    return load_nasa_k7()


@pytest.fixture(scope="module")
def pool(trellis):
    return build_codeword_pool(trellis, 200, seed=0)


@pytest.fixture(scope="module")
def fixed_batch(pool):
    """One fixed batch at high SNR, low interference."""
    device = torch.device("cpu")
    rng = np.random.default_rng(42)
    rx, bits = generate_training_batch(
        pool, BATCH_SIZE,
        snr_range=(10.0, 10.0),
        inr_range=(0.0, 0.0),
        period_range=(16, 16),
        rng=rng, device=device,
    )
    return rx, bits


def test_overfit_single_batch(fixed_batch):
    """Train on one batch; loss must drop and accuracy must be high."""
    rx, bits = fixed_batch

    model = BiRNNDecoder(hidden_size=32, input_dim=INPUT_DIM, cell_type="GRU",
                         bidirectional=True)
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

    assert final_loss < 0.4, \
        f"Final loss {final_loss:.4f} > 0.4 — model failed to overfit"
    assert final_loss < initial_loss * 0.5, \
        f"Loss didn't drop enough: {initial_loss:.4f} -> {final_loss:.4f}"

    model.eval()
    with torch.no_grad():
        logits = model(rx)
        preds = (torch.sigmoid(logits) > 0.5).float()
        accuracy = (preds == bits).float().mean().item()

    print(f"Overfit test: bit accuracy = {accuracy:.4f}")
    assert accuracy > 0.85, \
        f"Bit accuracy {accuracy:.4f} < 0.85"


def test_loss_decreases(fixed_batch):
    """Loss should decrease over training."""
    rx, bits = fixed_batch

    model = BiRNNDecoder(hidden_size=32, input_dim=INPUT_DIM, cell_type="GRU",
                         bidirectional=True)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    model.train()
    for step in range(200):
        logits = model(rx)
        loss = criterion(logits, bits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first_avg = np.mean(losses[:20])
    last_avg = np.mean(losses[-20:])

    assert last_avg < first_avg * 0.8, \
        f"Loss didn't decrease: first_20={first_avg:.4f}, last_20={last_avg:.4f}"
