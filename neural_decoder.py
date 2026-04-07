"""
neural_decoder.py — Bidirectional GRU end-to-end neural decoder (N1)

Part of: EE597 Search-Designed Trellis Codes Project

Architecture per CLAUDE.md:
  - Input: r[t] at each of N=524 time steps, input_dim=1
  - RNN: Bidirectional GRU, h=32/dir, concat dim=64
  - Output: Linear(64,1) shared across all 524 positions + sigmoid
  - Bit selection: positions 0,2,4,...,510 -> 256 info bits
  - ~6593 trainable parameters
"""

from __future__ import annotations
import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

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
    # Architecture — CLAUDE.md: h=32/dir, input_dim=1, bidirectional
    hidden_size: int = 32
    input_dim: int = 1
    cell_type: str = "GRU"
    bidirectional: bool = True
    use_batchnorm: bool = False

    # Training — CLAUDE.md: Adam lr=1e-3, reduce to 1e-4 after 50 epochs
    lr: float = 1e-3
    lr_step_epoch: int = 50
    lr_step_factor: float = 0.1
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

    # Channel training ranges — CLAUDE.md
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

    Per CLAUDE.md:
      Input:  (batch, 524, 1) — each received BPSK symbol is one time step
      BiGRU:  h=32/dir -> concat -> (batch, 524, 64)
      Output: Linear(64,1) at each position -> sigmoid -> 524 probabilities
      Select: info bit positions 0,2,4,...,510 -> 256 info bit predictions
    """

    def __init__(
        self,
        hidden_size: int = 32,
        input_dim: int = 1,
        cell_type: str = "GRU",
        bidirectional: bool = True,
        use_batchnorm: bool = False,
        n_info: int = K_INFO,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.use_batchnorm = use_batchnorm
        self.n_info = n_info  # 256

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

        # Optional BatchNorm1d between GRU and output layer
        self.bn = nn.BatchNorm1d(self.rnn_out_dim) if use_batchnorm else None

        # Linear maps pooled GRU output -> n_info logits directly
        self.output_linear = nn.Linear(self.rnn_out_dim, n_info)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N_sym, input_dim) — received signal

        Returns:
            logits: (batch, K_INFO=256) — logits for all info bits
        """
        rnn_out, _ = self.rnn(x)          # (batch, N_sym, 2H)
        pooled = rnn_out.mean(dim=1)      # (batch, 2H)
        return self.output_linear(pooled) # (batch, K_INFO)


# ─────────────────────────────────────────────
# Pre-encoded codeword pool
# ─────────────────────────────────────────────
def _max_coded_len(trellis: Trellis) -> int:
    """Max coded length: 2 * (K_INFO + K-1) = 2*(256+6) = 524 for K=7."""
    constraint_len = int(np.log2(trellis.n_states)) + 1
    return 2 * (K_INFO + constraint_len - 1)


def build_codeword_pool(
    trellis: Trellis,
    pool_size: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Pre-encode a pool of codewords. Pads shorter outputs to max length (524).

    Returns:
        dict with 'info_bits' (pool_size, K_INFO) and 'symbols' (pool_size, 524)
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
    Generate a training batch from pre-encoded pool + fresh channel noise.

    Returns:
        received: (batch_size, 524, 1) float32 tensor on device
        info_bits: (batch_size, K_INFO) float32 tensor on device
    """
    pool_size = len(pool['info_bits'])
    indices = rng.integers(0, pool_size, batch_size)
    info_bits = pool['info_bits'][indices]    # (B, K_INFO)
    symbols = pool['symbols'][indices]        # (B, 524)
    n_sym = symbols.shape[1]

    snr_db = rng.uniform(snr_range[0], snr_range[1], batch_size)
    inr_db = rng.uniform(inr_range[0], inr_range[1], batch_size)
    periods = rng.integers(period_range[0], period_range[1] + 1, batch_size)
    phases = rng.uniform(0, 2 * np.pi, batch_size)

    received = np.empty_like(symbols)
    for i in range(batch_size):
        nv = noise_var_from_snr(snr_db[i])
        noise = rng.standard_normal(n_sym).astype(np.float32) * np.sqrt(nv)
        amp = amplitude_from_inr(inr_db[i], nv)
        interf = generate_interference(n_sym, amp, float(periods[i]), phases[i])
        received[i] = symbols[i] + noise + interf.astype(np.float32)

    # (B, 524, 1) — each symbol is one time step, input_dim=1
    received = received.reshape(batch_size, n_sym, 1)

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
) -> tuple[float, float]:
    """Compute BLER and bit accuracy on a validation set at fixed SNR/INR.

    Returns:
        (bler, bit_acc) — block error rate and mean bit accuracy
    """
    model.eval()
    rng = np.random.default_rng(seed)
    n_errors = 0
    n_done = 0
    total_bit_acc = 0.0

    while n_done < n_blocks:
        bs = min(batch_size, n_blocks - n_done)

        pool_size = len(pool['info_bits'])
        indices = rng.integers(0, pool_size, bs)
        info_bits = pool['info_bits'][indices]
        symbols = pool['symbols'][indices]
        n_sym = symbols.shape[1]

        received = np.empty_like(symbols)
        for i in range(bs):
            nv = noise_var_from_snr(snr_db)
            noise = rng.standard_normal(n_sym).astype(np.float32) * np.sqrt(nv)
            amp = amplitude_from_inr(inr_db, nv)
            period = rng.integers(8, 33)
            phase = rng.uniform(0, 2 * np.pi)
            interf = generate_interference(n_sym, amp, float(period), phase)
            received[i] = symbols[i] + noise + interf.astype(np.float32)

        rx_tensor = torch.tensor(
            received.reshape(bs, n_sym, 1), dtype=torch.float32, device=device,
        )

        logits = model(rx_tensor)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int8)

        for i in range(bs):
            if not np.array_equal(preds[i], info_bits[i]):
                n_errors += 1
            total_bit_acc += np.mean(preds[i] == info_bits[i])

        n_done += bs

    bler = n_errors / n_blocks
    bit_acc = total_bit_acc / n_blocks
    return bler, bit_acc


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train_model(config: TrainConfig, seed: int = 42, model: Optional[BiRNNDecoder] = None) -> BiRNNDecoder:
    """
    Train a BiRNNDecoder model.

    Args:
        config: training configuration
        seed: random seed
        model: optional pretrained model to continue training (for curriculum learning).
               If None, a fresh model is created from config.

    Returns:
        trained model (loaded from best checkpoint)
    """
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Training BiRNNDecoder ({config.cell_type}, h={config.hidden_size}) "
          f"on {device}")

    if model is None:
        model = BiRNNDecoder(
            hidden_size=config.hidden_size,
            input_dim=config.input_dim,
            cell_type=config.cell_type,
            bidirectional=config.bidirectional,
            use_batchnorm=config.use_batchnorm,
        ).to(device)
        print("  Initialized fresh model")
    else:
        model = model.to(device)
        print("  Resuming from pretrained model (curriculum)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Adam + StepLR: lr=1e-3 for first 50 epochs, then 1e-4
    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = StepLR(
        optimizer,
        step_size=config.lr_step_epoch,
        gamma=config.lr_step_factor,
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

    rng = np.random.default_rng(seed + 1)
    best_val_bit_acc = 0.0
    patience_counter = 0
    training_log = []

    print(f"Training for up to {config.n_epochs} epochs "
          f"({config.batches_per_epoch} batches/epoch, batch_size={config.batch_size})")
    print(f"LR schedule: {config.lr} for epochs 0-{config.lr_step_epoch-1}, "
          f"then {config.lr * config.lr_step_factor}")
    print("-" * 60)

    for epoch in range(config.n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_bit_acc = 0.0
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
            # torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                epoch_bit_acc += (preds == info_bits).float().mean().item()

        avg_loss = epoch_loss / config.batches_per_epoch
        avg_bit_acc = epoch_bit_acc / config.batches_per_epoch
        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Validation
        val_bler, val_bit_acc = validate_bler(
            model, trellis, pool,
            config.val_snr_db, config.val_inr_db, config.val_blocks,
            device, seed=99,
        )

        entry = {
            'epoch': epoch,
            'loss': avg_loss,
            'val_bler': val_bler,
            'val_bit_acc': val_bit_acc,
            'lr': current_lr,
            'time_s': epoch_time,
        }
        training_log.append(entry)

        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | bit_acc={avg_bit_acc:.4f} | "
                  f"val_bit_acc={val_bit_acc:.4f} | val_bler={val_bler:.4f} | "
                  f"lr={current_lr:.1e} | time={epoch_time:.1f}s")

        # Checkpoint — save when val bit accuracy improves
        if val_bit_acc > best_val_bit_acc:
            best_val_bit_acc = val_bit_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': asdict(config),
                'epoch': epoch,
                'val_bler': val_bler,
                'val_bit_acc': val_bit_acc,
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  -> New best! val_bit_acc={val_bit_acc:.4f} val_bler={val_bler:.4f} "
                  f"Saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch} (patience={config.patience})")
                break

    print("-" * 60)
    print(f"Training complete. Best val bit_acc: {best_val_bit_acc:.4f}")
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
    Returns decode_fn compatible with eval.py's estimate_bler.

    decode_fn(received, period, phase, snr_db, inr_db) -> ndarray(K_INFO,)
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
        max_len = 524  # 2 * (K_INFO + 6) for K=7
        if len(received) < max_len:
            padded = np.zeros(max_len, dtype=np.float32)
            padded[:len(received)] = received
            received = padded

        # (1, 524, 1) — each symbol is one time step
        x = received[:max_len].reshape(1, max_len, 1).astype(np.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=dev)

        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.sigmoid(logits)

        bits = (probs[0].cpu().numpy() > 0.5).astype(np.int8)
        return bits  # (K_INFO,)

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
# Smoke test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running smoke test for neural_decoder...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Per CLAUDE.md: h=32/dir, input_dim=1, 524 time steps
    model = BiRNNDecoder(hidden_size=32, input_dim=1, cell_type="GRU",
                         bidirectional=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  BiGRU h=32: {n_params:,} parameters")

    x = torch.randn(4, 524, 1, device=device)
    logits = model(x)
    assert logits.shape == (4, K_INFO), f"Output shape wrong: {logits.shape}"
    print(f"  Forward pass: input {x.shape} -> output {logits.shape}")

    model.eval()
    decode_fn = make_decoder_n1(model, device)
    rx = np.random.randn(524).astype(np.float32)
    decoded = decode_fn(rx, 16.0, 0.0, 5.0, 5.0)
    assert decoded.shape == (K_INFO,), f"Decoded shape wrong: {decoded.shape}"
    assert decoded.dtype == np.int8
    print(f"  Decode wrapper: input ({rx.shape[0]},) -> output {decoded.shape}")

    trellis = load_nasa_k7()
    pool = build_codeword_pool(trellis, 100, seed=0)
    assert pool['info_bits'].shape == (100, K_INFO)
    n_sym = pool['symbols'].shape[1]
    print(f"  Pool: {pool['info_bits'].shape}, symbols length = {n_sym}")

    rng = np.random.default_rng(42)
    rx_batch, bits_batch = generate_training_batch(
        pool, batch_size=8,
        snr_range=(0.0, 10.0), inr_range=(-5.0, 10.0),
        period_range=(8, 32), rng=rng, device=device,
    )
    assert rx_batch.shape == (8, n_sym, 1), f"Batch shape wrong: {rx_batch.shape}"
    assert bits_batch.shape == (8, K_INFO)
    print(f"  Batch: received {rx_batch.shape}, info_bits {bits_batch.shape}")

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    logits = model(rx_batch)
    loss = criterion(logits, bits_batch)
    loss.backward()
    optimizer.step()
    print(f"  Training step: loss = {loss.item():.4f}")

    print("PASS")
