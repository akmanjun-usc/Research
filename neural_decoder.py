"""
neural_decoder.py — Bidirectional GRU end-to-end neural decoder (N1)

Part of: EE597 Search-Designed Trellis Codes Project

Architecture: received signal (262, 2) -> BiGRU -> Linear -> 256 info bit probabilities
Training: on-the-fly channel generation with pre-encoded codeword pool
"""

from __future__ import annotations
import numpy as np
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from channel import (
    bpsk_modulate, noise_var_from_snr, amplitude_from_inr,
    generate_interference, K_INFO,
)
from trellis import load_nasa_k7, Trellis


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
N_CODED = 512
N_STATES = 64
RATE = 0.5
RESULTS_DIR = Path(__file__).parent / "results" / "phase2a"


# ─────────────────────────────────────────────
# Training configuration
# ─────────────────────────────────────────────
@dataclass
class TrainConfig:
    """Configuration for BiGRU decoder training."""
    # Architecture
    hidden_size: int = 24
    input_dim: int = 2
    cell_type: str = "GRU"
    bidirectional: bool = True
    use_batchnorm: bool = False

    # Training
    lr: float = 1e-3
    lr_decay_patience: int = 10
    lr_factor: float = 0.1
    grad_clip: float = 1.0
    batch_size: int = 128
    batches_per_epoch: int = 1000
    n_epochs: int = 100
    pool_size: int = 50000

    # Validation
    val_blocks: int = 10000
    val_snr_db: float = 5.0
    val_inr_db: float = 5.0
    patience: int = 20

    # Channel training ranges
    snr_range: tuple[float, float] = (0.0, 10.0)
    inr_range: tuple[float, float] = (-5.0, 10.0)
    period_range: tuple[int, int] = (8, 32)

    # Paths and device
    checkpoint_dir: str = ""
    log_dir: str = ""
    device: str = ""

    def __post_init__(self) -> None:
        if not self.checkpoint_dir:
            self.checkpoint_dir = str(RESULTS_DIR / "checkpoints")
        if not self.log_dir:
            self.log_dir = str(RESULTS_DIR / "logs")
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# Model: Bidirectional RNN Decoder
# ─────────────────────────────────────────────
class BiRNNDecoder(nn.Module):
    """
    Bidirectional RNN decoder for convolutional codes.

    Takes received signal (batch, T, input_dim) and outputs logits
    for the first K_INFO=256 info bit positions.

    T = 262 trellis steps (256 info + 6 tail, each producing 2 coded bits).
    """

    def __init__(
        self,
        hidden_size: int = 24,
        input_dim: int = 2,
        cell_type: str = "GRU",
        bidirectional: bool = True,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.use_batchnorm = use_batchnorm
        self.n_info = K_INFO  # 256

        num_directions = 2 if bidirectional else 1
        self.rnn_out_dim = hidden_size * num_directions

        # RNN layer
        rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}[cell_type]
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Optional batch normalization
        self.bn = nn.BatchNorm1d(self.rnn_out_dim) if use_batchnorm else None

        # Output projection: shared across all time steps
        self.output_linear = nn.Linear(self.rnn_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, input_dim) received signal reshaped to trellis steps

        Returns:
            logits: (batch, K_INFO) — logits for the 256 info bit positions
        """
        # RNN forward pass
        rnn_out, _ = self.rnn(x)  # (batch, T, rnn_out_dim)

        # Optional batch norm: transpose to (batch, features, T), apply, transpose back
        if self.bn is not None:
            rnn_out = self.bn(rnn_out.transpose(1, 2)).transpose(1, 2)

        # Linear projection at each time step
        logits = self.output_linear(rnn_out).squeeze(-1)  # (batch, T)

        # Return only the first K_INFO positions (discard tail bits)
        return logits[:, :self.n_info]


# ─────────────────────────────────────────────
# Pre-encoded codeword pool
# ─────────────────────────────────────────────
def _max_coded_len(trellis: Trellis) -> int:
    """Maximum coded length: rate-1/2, K_INFO info bits + (constraint_len - 1) tail bits."""
    # For K=7 code: 2 * (256 + 6) = 524
    # Termination may use fewer tail bits, but we pad to this max
    constraint_len = int(np.log2(trellis.n_states)) + 1
    return 2 * (K_INFO + constraint_len - 1)


def build_codeword_pool(
    trellis: Trellis,
    pool_size: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Pre-encode a pool of codewords to avoid encoding bottleneck during training.

    Encoded outputs vary in length due to termination. We pad shorter
    codewords with zero symbols (no energy) to a fixed max length.

    Returns:
        dict with 'info_bits' (pool_size, K_INFO) and 'symbols' (pool_size, max_coded_len)
    """
    rng = np.random.default_rng(seed)
    max_len = _max_coded_len(trellis)

    info_arr = np.empty((pool_size, K_INFO), dtype=np.int8)
    sym_arr = np.zeros((pool_size, max_len), dtype=np.float32)

    for i in range(pool_size):
        info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
        coded = trellis.encode(info_bits)
        symbols = bpsk_modulate(coded)
        info_arr[i] = info_bits
        sym_arr[i, :len(symbols)] = symbols.astype(np.float32)

        if (i + 1) % 10000 == 0:
            print(f"  Pool encoding: {i + 1}/{pool_size}")

    return {
        'info_bits': info_arr,
        'symbols': sym_arr,
    }


# ─────────────────────────────────────────────
# Batch generation
# ─────────────────────────────────────────────
def generate_training_batch(
    pool: dict[str, np.ndarray],
    batch_size: int,
    snr_range: tuple[float, float],
    inr_range: tuple[float, float],
    period_range: tuple[int, int],
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a training batch using the pre-encoded pool.

    Returns:
        received: (batch_size, T, 2) float32 tensor on device
        info_bits: (batch_size, K_INFO) float32 tensor on device
    """
    pool_size = len(pool['info_bits'])
    indices = rng.integers(0, pool_size, batch_size)
    info_bits = pool['info_bits'][indices]    # (B, K_INFO)
    symbols = pool['symbols'][indices]        # (B, N_sym)
    n_sym = symbols.shape[1]

    # Sample channel parameters per block
    snr_db = rng.uniform(snr_range[0], snr_range[1], batch_size)
    inr_db = rng.uniform(inr_range[0], inr_range[1], batch_size)
    periods = rng.integers(period_range[0], period_range[1] + 1, batch_size)
    phases = rng.uniform(0, 2 * np.pi, batch_size)

    # Apply channel per block
    received = np.empty_like(symbols)
    for i in range(batch_size):
        nv = noise_var_from_snr(snr_db[i])
        noise = rng.standard_normal(n_sym).astype(np.float32) * np.sqrt(nv)
        amp = amplitude_from_inr(inr_db[i], nv)
        interf = generate_interference(n_sym, amp, float(periods[i]), phases[i])
        received[i] = symbols[i] + noise + interf.astype(np.float32)

    # Reshape to (B, T, 2) — trellis step format
    T = n_sym // 2
    received = received.reshape(batch_size, T, 2)

    return (
        torch.tensor(received, dtype=torch.float32, device=device),
        torch.tensor(info_bits, dtype=torch.float32, device=device),
    )


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
@torch.no_grad()
def validate_bler(
    model: BiRNNDecoder,
    trellis: Trellis,
    pool: dict[str, np.ndarray],
    snr_db: float,
    inr_db: float,
    n_blocks: int,
    device: torch.device,
    seed: int = 99,
    batch_size: int = 256,
) -> float:
    """Compute BLER on a fixed validation set. Returns BLER as float."""
    model.eval()
    rng = np.random.default_rng(seed)
    n_errors = 0
    n_done = 0

    while n_done < n_blocks:
        bs = min(batch_size, n_blocks - n_done)

        # Generate validation batch at fixed SNR/INR
        pool_size = len(pool['info_bits'])
        indices = rng.integers(0, pool_size, bs)
        info_bits = pool['info_bits'][indices]
        symbols = pool['symbols'][indices]
        n_sym = symbols.shape[1]

        # Apply channel at fixed SNR/INR
        received = np.empty_like(symbols)
        for i in range(bs):
            nv = noise_var_from_snr(snr_db)
            noise = rng.standard_normal(n_sym).astype(np.float32) * np.sqrt(nv)
            amp = amplitude_from_inr(inr_db, nv)
            period = rng.integers(8, 33)
            phase = rng.uniform(0, 2 * np.pi)
            interf = generate_interference(n_sym, amp, float(period), phase)
            received[i] = symbols[i] + noise + interf.astype(np.float32)

        T = n_sym // 2
        rx_tensor = torch.tensor(
            received.reshape(bs, T, 2), dtype=torch.float32, device=device,
        )

        logits = model(rx_tensor)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int8)

        # Block error: any bit differs
        for i in range(bs):
            if not np.array_equal(preds[i], info_bits[i]):
                n_errors += 1

        n_done += bs

    return n_errors / n_blocks


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train_model(config: TrainConfig, seed: int = 42) -> BiRNNDecoder:
    """
    Train a BiRNNDecoder model.

    Args:
        config: training configuration
        seed: random seed

    Returns:
        trained model (loaded from best checkpoint)
    """
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Training BiRNNDecoder ({config.cell_type}, h={config.hidden_size}) "
          f"on {device}")

    # Build model
    model = BiRNNDecoder(
        hidden_size=config.hidden_size,
        input_dim=config.input_dim,
        cell_type=config.cell_type,
        bidirectional=config.bidirectional,
        use_batchnorm=config.use_batchnorm,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=config.lr_decay_patience,
        factor=config.lr_factor,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Build codeword pool
    trellis = load_nasa_k7()
    print("Building codeword pool...")
    pool = build_codeword_pool(trellis, config.pool_size, seed=seed)
    print(f"  Pool: {pool['info_bits'].shape[0]} codewords, "
          f"symbol length = {pool['symbols'].shape[1]}")

    # Directories
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"best_{config.cell_type.lower()}.pt"
    ckpt_path = ckpt_dir / ckpt_name
    log_path = log_dir / f"training_{config.cell_type.lower()}_h{config.hidden_size}.json"

    # Training state
    rng = np.random.default_rng(seed + 1)
    best_val_bler = 1.0
    patience_counter = 0
    training_log = []

    print(f"Training for up to {config.n_epochs} epochs "
          f"({config.batches_per_epoch} batches/epoch, batch_size={config.batch_size})")
    print("-" * 60)

    for epoch in range(config.n_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for _ in range(config.batches_per_epoch):
            received, info_bits = generate_training_batch(
                pool, config.batch_size,
                config.snr_range, config.inr_range, config.period_range,
                rng, device,
            )

            logits = model(received)
            loss = criterion(logits, info_bits)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / config.batches_per_epoch
        epoch_time = time.time() - t0

        # Validation
        val_bler = validate_bler(
            model, trellis, pool,
            config.val_snr_db, config.val_inr_db, config.val_blocks,
            device, seed=99,
        )

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_bler)

        # Log
        entry = {
            'epoch': epoch,
            'loss': avg_loss,
            'val_bler': val_bler,
            'lr': current_lr,
            'time_s': epoch_time,
        }
        training_log.append(entry)

        # Save log after each epoch
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | val_bler={val_bler:.4f} | "
              f"lr={current_lr:.1e} | time={epoch_time:.1f}s")

        # Checkpoint
        if val_bler < best_val_bler:
            best_val_bler = val_bler
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': asdict(config),
                'epoch': epoch,
                'val_bler': val_bler,
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  -> New best! Saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch} (patience={config.patience})")
                break

    print("-" * 60)
    print(f"Training complete. Best val BLER: {best_val_bler:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Log: {log_path}")

    # Load best model
    best = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best['model_state_dict'])
    return model


# ─────────────────────────────────────────────
# Inference wrapper (compatible with eval.py)
# ─────────────────────────────────────────────
def make_decoder_n1(
    model: BiRNNDecoder,
    device: str | torch.device = "cpu",
) -> callable:
    """
    Returns a decode function compatible with eval.py's estimate_bler.

    Signature: decode_fn(received, period, phase, snr_db, inr_db) -> ndarray(K_INFO,)
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    def decode_fn(
        received: np.ndarray,
        period: float,
        phase: float,
        snr_db: float,
        inr_db: float,
    ) -> np.ndarray:
        # Pad to fixed max length (524) if shorter, then reshape to (1, 262, 2)
        max_len = 524  # 2 * (K_INFO + 6)
        if len(received) < max_len:
            padded = np.zeros(max_len, dtype=np.float32)
            padded[:len(received)] = received
            received = padded
        T = max_len // 2  # 262
        x = received[:max_len].reshape(1, T, 2).astype(np.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=dev)

        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.sigmoid(logits)

        bits = (probs[0].cpu().numpy() > 0.5).astype(np.int8)
        return bits  # shape (K_INFO,)

    return decode_fn


# ─────────────────────────────────────────────
# Load a trained model from checkpoint
# ─────────────────────────────────────────────
def load_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> BiRNNDecoder:
    """Load a trained BiRNNDecoder from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt['config']
    model = BiRNNDecoder(
        hidden_size=cfg['hidden_size'],
        input_dim=cfg['input_dim'],
        cell_type=cfg['cell_type'],
        bidirectional=cfg['bidirectional'],
        use_batchnorm=cfg['use_batchnorm'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────
# Smoke test (run with: python neural_decoder.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running smoke test for neural_decoder...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Test model instantiation and forward pass
    model = BiRNNDecoder(hidden_size=24, input_dim=2, cell_type="GRU",
                         bidirectional=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  BiGRU h=24: {n_params:,} parameters")

    # Forward pass with random input (batch=4, T=262, d=2)
    x = torch.randn(4, 262, 2, device=device)
    logits = model(x)
    assert logits.shape == (4, K_INFO), f"Output shape wrong: {logits.shape}"
    print(f"  Forward pass: input {x.shape} -> output {logits.shape}")

    # Test decode wrapper
    model.eval()
    decode_fn = make_decoder_n1(model, device)
    rx = np.random.randn(524).astype(np.float32)
    decoded = decode_fn(rx, 16.0, 0.0, 5.0, 5.0)
    assert decoded.shape == (K_INFO,), f"Decoded shape wrong: {decoded.shape}"
    assert decoded.dtype == np.int8
    print(f"  Decode wrapper: input ({rx.shape[0]},) -> output {decoded.shape}")

    # Test pool building (tiny)
    trellis = load_nasa_k7()
    pool = build_codeword_pool(trellis, 100, seed=0)
    assert pool['info_bits'].shape == (100, K_INFO)
    n_sym = pool['symbols'].shape[1]
    print(f"  Pool: {pool['info_bits'].shape}, symbols length = {n_sym}")

    # Test batch generation
    rng = np.random.default_rng(42)
    rx_batch, bits_batch = generate_training_batch(
        pool, batch_size=8,
        snr_range=(0.0, 10.0), inr_range=(-5.0, 10.0),
        period_range=(8, 32), rng=rng, device=device,
    )
    T = n_sym // 2
    assert rx_batch.shape == (8, T, 2), f"Batch shape wrong: {rx_batch.shape}"
    assert bits_batch.shape == (8, K_INFO)
    print(f"  Batch: received {rx_batch.shape}, info_bits {bits_batch.shape}")

    # Test one training step
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    logits = model(rx_batch)
    loss = criterion(logits, bits_batch)
    loss.backward()
    optimizer.step()
    print(f"  Training step: loss = {loss.item():.4f}")

    print("PASS")
