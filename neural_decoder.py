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
import argparse
import numpy as np
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
    generate_interference, awgn_channel, K_INFO,
)
from trellis import load_nasa_k7, Trellis, CONSTRAINT_LEN


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
_N_TAIL = CONSTRAINT_LEN - 1           # 6 tail bits
_N_CODED_FIXED = 2 * (K_INFO + _N_TAIL)  # 524 coded bits
RESULTS_DIR = Path(__file__).parent / "results" / "phase2a"


# ─────────────────────────────────────────────
# Fixed-tail encoder helper
# ─────────────────────────────────────────────

def _encode_fixed_tail(info_bits: np.ndarray, trellis: Trellis) -> np.ndarray:
    """
    Encode info_bits with exactly _N_TAIL = 6 zero-input tail steps.

    Unlike trellis.encode(), which stops early when the state reaches 0,
    this function always produces exactly 2*(K_INFO + 6) = 524 coded bits.

    Returns:
        coded_bits: (524,) int8
    """
    state = 0
    coded: list[int] = []
    for bit in info_bits:
        coded.extend(trellis.output_bits[state, int(bit)])
        state = trellis.next_state[state, int(bit)]
    for _ in range(_N_TAIL):
        coded.extend(trellis.output_bits[state, 0])
        state = trellis.next_state[state, 0]
    return np.array(coded, dtype=np.int8)


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

    # Training
    lr: float = 1e-3
    lr_step_epoch: int = 50
    lr_step_factor: float = 1.0
    batch_size: int = 128
    batches_per_epoch: int = 200
    n_epochs: int = 200

    # Validation — every val_every epochs; patience is in validation rounds
    val_blocks: int = 1000
    val_every: int = 5
    val_snr_db: float = 5.0
    val_inr_db: float = 5.0
    patience: int = 20  # validation rounds without BLER improvement

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
# Model: Bidirectional RNN Decoder (final hidden state pooling)
# ─────────────────────────────────────────────
class BiRNNDecoder(nn.Module):
    """
    Bidirectional RNN decoder for convolutional codes.

    Per CLAUDE.md:
      Input:  (batch, 524, 1) — each received BPSK symbol is one time step
      BiGRU:  h=32/dir -> concat -> (batch, 524, 64)
      Output: Linear(64, K_INFO) on pooled final hidden states -> (batch, 256)
    """

    def __init__(
        self,
        hidden_size: int = 32,
        input_dim: int = 1,
        cell_type: str = "GRU",
        bidirectional: bool = True,
        n_info: int = K_INFO,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.n_info = n_info  # 256

        num_directions = 2 if bidirectional else 1
        self.rnn_out_dim = hidden_size * num_directions

        rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}[cell_type]
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.output_linear = nn.Linear(self.rnn_out_dim, n_info)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N_sym, input_dim) — received signal

        Returns:
            logits: (batch, K_INFO=256)
        """
        _, h_n = self.rnn(x)  # h_n: (2, B, H) for bidirectional
        pooled = h_n.transpose(0, 1).reshape(x.size(0), -1)  # (B, 2H)
        return self.output_linear(pooled)  # (batch, K_INFO)


# ─────────────────────────────────────────────
# Batch generation (on-the-fly)
# ─────────────────────────────────────────────
def generate_training_batch(
    trellis: Trellis,
    batch_size: int,
    snr_db: float,
    inr_db: float,
    period_range: tuple[int, int],
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a training batch on-the-fly.

    SNR and INR are fixed for the whole batch; period and phase are drawn
    independently per block to match the evaluation distribution.

    Returns:
        received: (batch_size, 524, 1) float32 tensor on device
        info_bits: (batch_size, K_INFO) float32 tensor on device
    """
    noise_var = noise_var_from_snr(snr_db)
    amplitude = amplitude_from_inr(inr_db, noise_var)

    received_list: list[np.ndarray] = []
    info_bits_list: list[np.ndarray] = []

    for _ in range(batch_size):
        period = float(rng.integers(period_range[0], period_range[1] + 1))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))

        info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
        coded = _encode_fixed_tail(info_bits, trellis)  # always 524 bits
        symbols = bpsk_modulate(coded)
        N = len(symbols)

        noise = rng.standard_normal(N).astype(np.float32) * np.sqrt(noise_var)
        interf = generate_interference(N, amplitude, period, phase).astype(np.float32)
        received = symbols + noise + interf

        received_list.append(received)
        info_bits_list.append(info_bits)

    received_arr = np.stack(received_list).reshape(batch_size, _N_CODED_FIXED, 1)
    info_arr = np.stack(info_bits_list).astype(np.float32)

    return (
        torch.tensor(received_arr, dtype=torch.float32, device=device),
        torch.tensor(info_arr, dtype=torch.float32, device=device),
    )


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
@torch.no_grad()
def validate_bler(
    model: BiRNNDecoder,
    trellis: Trellis,
    snr_db: float,
    inr_db: float,
    n_blocks: int,
    device: torch.device,
    seed: int = 99,
    period_range: tuple[int, int] = (8, 32),
    batch_size: int = 256,
) -> float:
    """
    Compute BLER at fixed SNR/INR with random period/phase per block.

    Returns:
        bler — block error rate
    """
    model.eval()
    rng = np.random.default_rng(seed)
    noise_var = noise_var_from_snr(snr_db)
    amplitude = amplitude_from_inr(inr_db, noise_var)

    n_errors = 0
    n_done = 0

    while n_done < n_blocks:
        bs = min(batch_size, n_blocks - n_done)

        received_list: list[np.ndarray] = []
        info_bits_list: list[np.ndarray] = []

        for _ in range(bs):
            period = float(rng.integers(period_range[0], period_range[1] + 1))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))

            info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
            coded = _encode_fixed_tail(info_bits, trellis)  # always 524 bits
            symbols = bpsk_modulate(coded)
            N = len(symbols)

            noise = rng.standard_normal(N).astype(np.float32) * np.sqrt(noise_var)
            interf = generate_interference(N, amplitude, period, phase).astype(np.float32)
            received = symbols + noise + interf

            received_list.append(received)
            info_bits_list.append(info_bits)

        received_arr = np.stack(received_list).reshape(bs, _N_CODED_FIXED, 1)
        rx_tensor = torch.tensor(received_arr, dtype=torch.float32, device=device)

        logits = model(rx_tensor)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int8)

        for i in range(bs):
            if not np.array_equal(preds[i], info_bits_list[i]):
                n_errors += 1

        n_done += bs

    model.train()
    return n_errors / n_blocks


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train_model(config: TrainConfig, seed: int = 42, model: Optional[BiRNNDecoder] = None) -> BiRNNDecoder:
    """
    Train a BiRNNDecoder model.

    Args:
        config: training configuration
        seed:   random seed
        model:  optional pretrained model to continue training (curriculum learning).
                If None, a fresh model is created from config.

    Returns:
        trained model (loaded from best checkpoint by val BLER)
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
        ).to(device)
        print("  Initialized fresh model")
    else:
        model = model.to(device)
        print("  Resuming from pretrained model (curriculum)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = StepLR(
        optimizer,
        step_size=config.lr_step_epoch,
        gamma=config.lr_step_factor,
    )
    criterion = nn.BCEWithLogitsLoss()

    trellis = load_nasa_k7()

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    seed_ckpt_path = ckpt_dir / f"best_{config.cell_type.lower()}_seed{seed}.pt"
    history_path = log_dir / f"history_seed{seed}.npz"

    rng = np.random.default_rng(seed + 1)
    best_val_bler = float('inf')
    best_epoch = -1
    patience_counter = 0
    best_model_state = None
    history: dict[str, list[float | int]] = {
        'train_loss': [],
        'train_bit_acc': [],
        'lr': [],
        'epoch_time_s': [],
        'val_bler': [],
        'val_epoch': [],
    }

    print(f"Training for up to {config.n_epochs} epochs "
          f"({config.batches_per_epoch} batches/epoch, batch_size={config.batch_size})")
    print(f"Validation every {config.val_every} epochs, "
          f"patience={config.patience} validation rounds")
    print("-" * 60)

    for epoch in range(config.n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct_bits = 0
        epoch_total_bits = 0
        t0 = time.time()

        for _ in range(config.batches_per_epoch):
            # Draw SNR/INR once per batch; period/phase vary per block inside
            snr_db = float(rng.uniform(config.snr_range[0], config.snr_range[1]))
            inr_db = float(rng.uniform(config.inr_range[0], config.inr_range[1]))

            received, info_bits = generate_training_batch(
                trellis, config.batch_size, snr_db, inr_db,
                config.period_range, rng, device,
            )

            logits = model(received)
            loss = criterion(logits, info_bits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5)
                epoch_correct_bits += (preds == (info_bits > 0.5)).sum().item()
                epoch_total_bits += info_bits.numel()

        avg_loss = epoch_loss / config.batches_per_epoch
        train_bit_acc = epoch_correct_bits / epoch_total_bits if epoch_total_bits else 0.0
        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        history['train_loss'].append(float(avg_loss))
        history['train_bit_acc'].append(float(train_bit_acc))
        history['lr'].append(float(current_lr))
        history['epoch_time_s'].append(float(epoch_time))

        # Validate every val_every epochs (and always on the final epoch)
        if (epoch + 1) % config.val_every == 0 or epoch == config.n_epochs - 1:
            val_bler = validate_bler(
                model, trellis,
                config.val_snr_db, config.val_inr_db, config.val_blocks,
                device, seed=99, period_range=config.period_range,
            )
            history['val_bler'].append(float(val_bler))
            history['val_epoch'].append(int(epoch + 1))

            print(f"Epoch {epoch+1:3d} | train_loss={avg_loss:.4f} | val_bler={val_bler:.4f}")
            print(f"  [train] bit_acc={train_bit_acc:.4f}  lr={current_lr:.1e}  "
                  f"time={epoch_time:.1f}s")

            if val_bler < best_val_bler:
                best_val_bler = val_bler
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'config': asdict(config),
                    'epoch': int(epoch + 1),
                    'val_bler': float(val_bler),
                    'loss': float(avg_loss),
                    'seed': int(seed),
                }
                torch.save(checkpoint, seed_ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping at epoch {epoch+1} "
                          f"(patience={config.patience} rounds without improvement)")
                    break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    np.savez(
        history_path,
        train_loss=np.array(history['train_loss'], dtype=np.float32),
        train_bit_acc=np.array(history['train_bit_acc'], dtype=np.float32),
        lr=np.array(history['lr'], dtype=np.float32),
        epoch_time_s=np.array(history['epoch_time_s'], dtype=np.float32),
        val_bler=np.array(history['val_bler'], dtype=np.float32),
        val_epoch=np.array(history['val_epoch'], dtype=np.int32),
    )

    print("-" * 60)
    print(f"Training complete. Best val BLER: {best_val_bler:.4f} (epoch {best_epoch})")
    print(f"Seed checkpoint: {seed_ckpt_path}")
    print(f"History saved to {history_path}")
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
        if len(received) < _N_CODED_FIXED:
            padded = np.zeros(_N_CODED_FIXED, dtype=np.float32)
            padded[:len(received)] = received
            received = padded

        x = received[:_N_CODED_FIXED].reshape(1, _N_CODED_FIXED, 1).astype(np.float32)
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
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────
# CLI / smoke test
# ─────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train or smoke-test the Phase 2a neural decoder.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="train the decoder instead of running the smoke test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed used for training (default: 42)",
    )
    return parser.parse_args()


def _run_smoke_test() -> None:
    """Run a lightweight sanity check for the decoder module."""
    print("Running smoke test for neural_decoder...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Per CLAUDE.md: h=32/dir, input_dim=1, 524 time steps
    model = BiRNNDecoder(hidden_size=32, input_dim=1, cell_type="GRU",
                         bidirectional=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  BiGRU h=32: {n_params:,} parameters")

    x = torch.randn(4, _N_CODED_FIXED, 1, device=device)
    logits = model(x)
    assert logits.shape == (4, K_INFO), f"Output shape wrong: {logits.shape}"
    print(f"  Forward pass: input {x.shape} -> output {logits.shape}")

    model.eval()
    decode_fn = make_decoder_n1(model, device)
    rx = np.random.randn(_N_CODED_FIXED).astype(np.float32)
    decoded = decode_fn(rx, 16.0, 0.0, 5.0, 5.0)
    assert decoded.shape == (K_INFO,), f"Decoded shape wrong: {decoded.shape}"
    assert decoded.dtype == np.int8
    print(f"  Decode wrapper: input ({rx.shape[0]},) -> output {decoded.shape}")

    trellis = load_nasa_k7()
    rng = np.random.default_rng(42)

    info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
    coded = _encode_fixed_tail(info_bits, trellis)
    assert len(coded) == _N_CODED_FIXED, f"Coded length wrong: {len(coded)}"
    print(f"  _encode_fixed_tail: {K_INFO} info bits -> {len(coded)} coded bits")

    rx_batch, bits_batch = generate_training_batch(
        trellis, batch_size=8, snr_db=5.0, inr_db=5.0,
        period_range=(8, 32), rng=rng, device=device,
    )
    assert rx_batch.shape == (8, _N_CODED_FIXED, 1), f"Batch shape wrong: {rx_batch.shape}"
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

    val_bler = validate_bler(
        model, trellis, snr_db=5.0, inr_db=5.0,
        n_blocks=20, device=device, seed=0,
    )
    print(f"  Validation BLER (random model, 20 blocks): {val_bler:.4f}")

    print("PASS")


if __name__ == "__main__":
    args = _parse_args()

    if args.train:
        config = TrainConfig()
        train_model(config, seed=args.seed)
    else:
        _run_smoke_test()
