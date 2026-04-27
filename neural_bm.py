"""
neural_bm.py — Neural Branch Metric Estimator (N2)

Phase 2b: BiGRU branch metric estimator + Viterbi integration.

Architecture:
  Input:   (batch, T, 2)  — T=262 trellis steps, 2 received samples per step
  BiGRU:   h=16/dir, 1 layer → (batch, T, 32)
  Linear(32, 4)             → (batch, T, 4)  branch metric scores

The 4 output channels correspond to the 4 possible BPSK output pairs:
  index 0: coded bits (1,1) → BPSK (-1, -1)
  index 1: coded bits (1,0) → BPSK (-1, +1)
  index 2: coded bits (0,1) → BPSK (+1, -1)
  index 3: coded bits (0,0) → BPSK (+1, +1)

Part of: EE597 Search-Designed Trellis Codes Project
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from channel import (
    amplitude_from_inr,
    awgn_channel,
    bpsk_modulate,
    generate_interference,
    noise_var_from_snr,
)
from decoders import _build_reverse_trellis
from trellis import CONSTRAINT_LEN, K_INFO, Trellis, load_nasa_k7

# Number of memory elements in the K=7 code (= constraint length − 1)
_N_TAIL = CONSTRAINT_LEN - 1   # 6
# Fixed coded block length (always 2 × (K_INFO + N_TAIL) = 524)
_N_CODED_FIXED = 2 * (K_INFO + _N_TAIL)  # 524


# ─────────────────────────────────────────────
# Fixed-tail encoder helper
# ─────────────────────────────────────────────

def _encode_fixed_tail(info_bits: np.ndarray, trellis: Trellis) -> np.ndarray:
    """
    Encode info_bits with exactly _N_TAIL = 6 zero-input tail steps.

    Unlike trellis.encode(), which stops early when the state reaches 0,
    this function always produces exactly 2*(K_INFO + 6) = 524 coded bits.
    For the NASA K=7 shift-register code, 6 zero inputs always drive the
    state to 0 regardless of the starting state, so the output is valid.

    Returns:
        coded_bits: (524,) int8
    """
    from phase3_native import encode_native

    return encode_native(
        trellis.next_state,
        trellis.output_bits,
        np.asarray(info_bits, dtype=np.int8),
        trellis.n_states,
        K_INFO,
        _N_TAIL,
    )


# ─────────────────────────────────────────────
# BPSK pair constants
# ─────────────────────────────────────────────

# 4 possible BPSK output pairs for a rate-1/2 code.
# Row j = BPSK symbols for output pair index j.
# BPSK convention: bit 0 → +1, bit 1 → −1  (x = 1 − 2·bit)
BPSK_PAIRS: np.ndarray = np.array(
    [
        [-1.0, -1.0],  # index 0: coded bits (1, 1)
        [-1.0, +1.0],  # index 1: coded bits (1, 0)
        [+1.0, -1.0],  # index 2: coded bits (0, 1)
        [+1.0, +1.0],  # index 3: coded bits (0, 0)
    ],
    dtype=np.float64,
)


# ─────────────────────────────────────────────
# BPSK pair → index mapping
# ─────────────────────────────────────────────

def output_bits_to_index(c0: int, c1: int) -> int:
    """
    Map output bit pair (c0, c1) ∈ {0,1}² to the BPSK_PAIRS row index.

    BPSK: bit 0 → +1, bit 1 → −1  (x = 1 − 2·bit)
    """
    x0 = 1 - 2 * c0
    x1 = 1 - 2 * c1
    for i, pair in enumerate(BPSK_PAIRS):
        if pair[0] == x0 and pair[1] == x1:
            return i
    raise ValueError(f"No BPSK pair found for bits ({c0}, {c1})")


def build_branch_output_index(trellis: Trellis) -> np.ndarray:
    """
    Precompute mapping from (state, input_bit) → BPSK_PAIRS index.

    For each trellis transition (s, b), look up which BPSK output pair it
    produces and record the index into BPSK_PAIRS.  Called once per trellis;
    reused at every Viterbi step.

    Returns:
        index_table: (S, 2) int32
    """
    S = trellis.n_states
    index_table = np.zeros((S, 2), dtype=np.int32)
    for s in range(S):
        for b in range(2):
            c0 = int(trellis.output_bits[s, b, 0])
            c1 = int(trellis.output_bits[s, b, 1])
            index_table[s, b] = output_bits_to_index(c0, c1)
    return index_table


# ─────────────────────────────────────────────
# NeuralBranchMetric module
# ─────────────────────────────────────────────

class NeuralBranchMetric(nn.Module):
    """
    BiGRU-based branch metric estimator for rate-1/2 convolutional codes.

    Input:  (batch, T, 2)  — paired received samples, one pair per trellis step
    Output: (batch, T, num_branch_outputs)  — learned log-likelihood scores;
            output[b, t, j] estimates the branch metric for BPSK hypothesis j
            at trellis step t of block b.
    """

    def __init__(
        self,
        hidden_size: int = 16,
        num_layers: int = 1,
        num_branch_outputs: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_branch_outputs = num_branch_outputs

        self.bigru = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Batch norm disabled; keep an identity layer so the interface stays stable.
        self.bn = nn.Identity()
        self.linear = nn.Linear(2 * hidden_size, num_branch_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, 2) float32

        Returns:
            out: (batch, T, num_branch_outputs) float32
        """
        # BiGRU: (batch, T, 2) → (batch, T, 2*h)
        gru_out, _ = self.bigru(x)

        bn_out = self.bn(gru_out)  # (batch, T, 2*h)

        # Linear head per time step: (batch, T, num_branch_outputs)
        return self.linear(bn_out)


# ─────────────────────────────────────────────
# Signal pairing utility
# ─────────────────────────────────────────────

def pair_received_signal(y_flat: np.ndarray) -> np.ndarray:
    """
    Reshape flat received signal into trellis-step pairs.

    Consecutive samples (y[2t], y[2t+1]) belong to trellis step t for a
    rate-1/2 code.

    Args:
        y_flat: (N,) or (batch, N) float array

    Returns:
        y_paired: (T, 2) or (batch, T, 2) where T = N // 2
    """
    if y_flat.ndim == 1:
        T = len(y_flat) // 2
        return y_flat.reshape(T, 2)
    if y_flat.ndim == 2:
        batch, N = y_flat.shape
        T = N // 2
        return y_flat.reshape(batch, T, 2)
    raise ValueError(f"Expected 1-D or 2-D array, got shape {y_flat.shape}")


# ─────────────────────────────────────────────
# Oracle branch metric computation
# ─────────────────────────────────────────────

def compute_oracle_metrics(
    y_paired: torch.Tensor,
    interference_paired: torch.Tensor,
) -> torch.Tensor:
    """
    Compute unnormalized oracle branch metrics for all 4 BPSK hypotheses.

    For hypothesis h_j ∈ BPSK_PAIRS:
        oracle[b, t, j] = −‖ y[b,t] − h_j − interf[b,t] ‖²

    Returns unnormalized negative squared distances. The σ² scaling is applied
    externally at inference time (deterministic from SNR) rather than baked into
    the regression target — this keeps training targets on a consistent scale
    across the full SNR training distribution.

    Args:
        y_paired:            (batch, T, 2) float32
        interference_paired: (batch, T, 2) float32  — true interference

    Returns:
        oracle_metrics: (batch, T, 4) float32
    """
    device = y_paired.device
    # BPSK hypotheses tensor: (4, 2)
    pairs = torch.tensor(BPSK_PAIRS, dtype=torch.float32, device=device)

    # Effective received after removing interference: (batch, T, 2)
    r = y_paired - interference_paired

    # Squared distance to each hypothesis: (batch, T, 4, 2) → (batch, T, 4)
    # r: (batch, T, 1, 2) − pairs: (1, 1, 4, 2) → (batch, T, 4, 2)
    diff = r.unsqueeze(2) - pairs[None, None, :, :]
    sq_norm = (diff * diff).sum(dim=-1)  # (batch, T, 4)

    return -sq_norm


# ─────────────────────────────────────────────
# Viterbi with pre-computed neural branch metrics
# ─────────────────────────────────────────────

def viterbi_neural_bm(
    branch_metrics: np.ndarray,
    trellis: Trellis,
    index_table: np.ndarray,
) -> np.ndarray:
    """
    Standard Viterbi ACS + traceback using pre-computed neural branch metrics.

    The NN forward pass is run ONCE per block (outside this function), producing
    all T×4 metrics.  This function performs the DP search given those metrics.

    Args:
        branch_metrics: (T, 4) float64 — NN output for one block (single batch element)
        trellis:        Trellis object (NASA K=7 or custom)
        index_table:    (S, 2) int32 — from build_branch_output_index()

    Returns:
        decoded_bits: (K_INFO,) int8
    """
    INF = -1e18
    S = trellis.n_states
    T = len(branch_metrics)

    # Build (and cache) reverse trellis
    if not hasattr(trellis, '_rev_cache'):
        trellis._rev_cache = _build_reverse_trellis(trellis)
    rev = trellis._rev_cache

    rev_src: np.ndarray = rev['rev_src']      # (S, max_inc)  int32
    rev_bit: np.ndarray = rev['rev_bit']      # (S, max_inc)  int8
    n_incoming: np.ndarray = rev['n_incoming']  # (S,)         int32
    max_inc: int = rev['max_incoming']

    # Validity mask: True where the branch slot is padding (no actual branch)
    inc_mask = np.arange(max_inc)[np.newaxis, :] >= n_incoming[:, np.newaxis]  # (S, max_inc)

    pm_cur = np.full(S, INF, dtype=np.float64)
    pm_cur[0] = 0.0

    backtrack = np.zeros((T, S), dtype=np.int32)
    input_bits_table = np.zeros((T, S), dtype=np.int8)

    rev_src_i32 = rev_src.astype(np.int32)
    rev_bit_i32 = rev_bit.astype(np.int32)

    for t in range(T):
        # Output-pair index for every (dest, incoming) slot: (S, max_inc)
        bm_idx = index_table[rev_src_i32, rev_bit_i32]

        # Branch metrics from NN lookup: (S, max_inc)
        bm = branch_metrics[t, bm_idx]

        # Total path metric: predecessor + branch metric
        total = pm_cur[rev_src_i32] + bm  # (S, max_inc)
        total[inc_mask] = INF             # mask padding slots

        # ACS: best incoming branch per destination state
        best_idx = np.argmax(total, axis=1)       # (S,)
        pm_cur = total[np.arange(S), best_idx]
        backtrack[t] = rev_src_i32[np.arange(S), best_idx]
        input_bits_table[t] = rev_bit[np.arange(S), best_idx]

    # Traceback from state 0 (terminated trellis)
    decoded = np.zeros(T, dtype=np.int8)
    state = 0
    for t in range(T - 1, -1, -1):
        decoded[t] = input_bits_table[t, state]
        state = backtrack[t, state]

    return decoded[:K_INFO]


# ─────────────────────────────────────────────
# Inference wrapper (eval.py compatible)
# ─────────────────────────────────────────────

def make_decoder_n2(
    model: NeuralBranchMetric,
    device: str | torch.device,
    trellis: Trellis,
    index_table: Optional[np.ndarray] = None,
) -> callable:
    """
    Create an N2 decode function compatible with eval.py's estimate_bler.

    Signature of returned function:
        decode_fn(received, period, phase, snr_db, inr_db) → (K_INFO,) int8

    Args:
        model:       trained NeuralBranchMetric (moved to device internally)
        device:      torch device string or object
        trellis:     Trellis object
        index_table: (S, 2) int32, computed if None

    Returns:
        decode_fn callable
    """
    if index_table is None:
        index_table = build_branch_output_index(trellis)

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
        # Pair: (N,) → (1, T, 2) tensor on device
        y_paired = pair_received_signal(received)  # (T, 2)
        x = torch.tensor(y_paired, dtype=torch.float32, device=dev).unsqueeze(0)

        with torch.no_grad():
            nn_out = model(x)  # (1, T, 4)

        bm = nn_out.squeeze(0).cpu().numpy()  # (T, 4)
        noise_var = noise_var_from_snr(snr_db)
        bm = bm / (2.0 * float(noise_var))
        return viterbi_neural_bm(bm, trellis, index_table)

    return decode_fn


# ─────────────────────────────────────────────
# Batch generation helper (training)
# ─────────────────────────────────────────────

def _generate_training_batch(
    batch_size: int,
    snr_db: float,
    inr_db: float,
    period_range: tuple[int, int],
    trellis: Trellis,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate one training batch on-the-fly.

    SNR and INR are shared across the batch (noise power and interference
    amplitude are fixed). Period and phase are drawn independently per block
    to match the evaluation distribution.

    Returns:
        y_paired:       (batch, T, 2) float32 tensor — received samples paired
        oracle_metrics: (batch, T, 4) float32 tensor — ground-truth branch metrics
    """
    noise_var = noise_var_from_snr(snr_db)
    noise_sigma = float(np.sqrt(noise_var))
    amplitude = amplitude_from_inr(inr_db, noise_var)

    y_list: list[np.ndarray] = []
    interf_list: list[np.ndarray] = []

    for _ in range(batch_size):
        period = float(rng.integers(period_range[0], period_range[1] + 1))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))

        info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
        coded = _encode_fixed_tail(info_bits, trellis)   # always 524 bits
        symbols = bpsk_modulate(coded)
        N = len(symbols)  # always 524

        received = awgn_channel(symbols, snr_db, inr_db, period, phase, rng)
        interference = generate_interference(N, amplitude, period, phase)

        y_list.append(pair_received_signal(received))      # (262, 2)
        interf_list.append(pair_received_signal(interference))  # (262, 2)

    y_batch = torch.tensor(np.stack(y_list), dtype=torch.float32, device=device)
    interf_batch = torch.tensor(np.stack(interf_list), dtype=torch.float32, device=device)
    oracle = compute_oracle_metrics(y_batch, interf_batch)

    return y_batch, oracle


# ─────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────

def _validate_bler(
    model: NeuralBranchMetric,
    trellis: Trellis,
    index_table: np.ndarray,
    config: dict,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    """Run BLER validation at fixed (val_snr, val_inr) with random P, φ."""
    model.eval()
    n_errors = 0
    val_snr: float = config['val_snr']
    val_inr: float = config['val_inr']
    val_blocks: int = config['val_blocks']
    period_range: tuple[int, int] = config['period_range']

    with torch.no_grad():
        for _ in range(val_blocks):
            period = float(rng.integers(period_range[0], period_range[1] + 1))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))

            info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
            coded = _encode_fixed_tail(info_bits, trellis)   # always 524 bits
            symbols = bpsk_modulate(coded)
            received = awgn_channel(symbols, val_snr, val_inr, period, phase, rng)

            y_paired = pair_received_signal(received)   # (T, 2)
            x = torch.tensor(y_paired, dtype=torch.float32, device=device).unsqueeze(0)
            bm = model(x).squeeze(0).cpu().numpy()      # (T, 4)
            noise_var = noise_var_from_snr(val_snr)
            bm = bm / (2.0 * float(noise_var))

            decoded = viterbi_neural_bm(bm, trellis, index_table)
            if not np.array_equal(decoded, info_bits):
                n_errors += 1

    model.train()
    return n_errors / val_blocks


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    'hidden_size': 16,
    'num_layers': 1,
    'lr': 1e-3,
    'lr_decay_factor': 1.0, # no decay by default
    'lr_decay_every': 50,
    'batch_size': 256,
    'batches_per_epoch': 200,
    'num_epochs': 200,
    'patience': 20,
    'snr_range': (0.0, 10.0),
    'inr_range': (-5.0, 15.0),
    'period_range': (8, 32),
    'val_snr': 5.0,
    'val_inr': 5.0,
    'val_blocks': 1000,
    'val_every': 5,
    'checkpoint_dir': 'results/phase2b/checkpoints',
    'device': 'cpu',
}


def train_neural_bm(config: dict, seed: int = 42) -> NeuralBranchMetric:
    """
    Train NeuralBranchMetric with on-the-fly batch generation.

    Training targets: oracle branch metrics (MSE loss).
    Validation metric: block error rate (BLER) via viterbi_neural_bm.

    Args:
        config: training configuration (see DEFAULT_CONFIG for keys)
        seed:   random seed

    Returns:
        Best model by validation BLER (also saved to checkpoint_dir)
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(0, 2**31)))

    device = torch.device(config.get('device', 'cpu'))
    trellis = load_nasa_k7()
    index_table = build_branch_output_index(trellis)

    model = NeuralBranchMetric(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = StepLR(
        optimizer,
        step_size=config['lr_decay_every'],
        gamma=config['lr_decay_factor'],
    )
    criterion = nn.MSELoss()

    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    batches_per_epoch: int = config.get('batches_per_epoch', 10)
    best_val_bler = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    history: dict = {
        'train_mse': [],          # one value per epoch
        'train_mse_h0': [],       # per-hypothesis, one value per epoch
        'train_mse_h1': [],
        'train_mse_h2': [],
        'train_mse_h3': [],
        'val_bler': [],           # one value per validation epoch
        'val_epoch': [],          # which epoch the val_bler was recorded at
    }

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_mse = 0.0
        epoch_mse_per_hyp = np.zeros(4, dtype=np.float64)

        for batch_idx in range(batches_per_epoch):
            # Sample channel params fresh for each mini-batch
            snr_db = float(rng.uniform(config['snr_range'][0], config['snr_range'][1]))
            inr_db = float(rng.uniform(config['inr_range'][0], config['inr_range'][1]))

            y_batch, oracle = _generate_training_batch(
                config['batch_size'], snr_db, inr_db, config['period_range'],
                trellis, rng, device,
            )

            if epoch == 0 and batch_idx == 0:
                with torch.no_grad():
                    print(
                        f"  [diagnostic] oracle metrics — "
                        f"mean={oracle.mean():.3f}  std={oracle.std():.3f}  "
                        f"min={oracle.min():.3f}  max={oracle.max():.3f}"
                    )

            pred = model(y_batch)            # (batch, T, 4)
            loss = criterion(pred, oracle)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_mse += loss.item()
            with torch.no_grad():
                per_hyp = ((pred - oracle) ** 2).mean(dim=(0, 1))  # (4,)
                epoch_mse_per_hyp += per_hyp.cpu().numpy()

        epoch_mse /= batches_per_epoch
        epoch_mse_per_hyp /= batches_per_epoch
        scheduler.step()

        history['train_mse'].append(epoch_mse)
        history['train_mse_h0'].append(float(epoch_mse_per_hyp[0]))
        history['train_mse_h1'].append(float(epoch_mse_per_hyp[1]))
        history['train_mse_h2'].append(float(epoch_mse_per_hyp[2]))
        history['train_mse_h3'].append(float(epoch_mse_per_hyp[3]))

        # Periodic BLER validation
        if (epoch + 1) % config['val_every'] == 0 or epoch == config['num_epochs'] - 1:
            val_bler = _validate_bler(model, trellis, index_table, config, rng, device)
            history['val_bler'].append(val_bler)
            history['val_epoch'].append(epoch + 1)

            print(f"Epoch {epoch+1:3d} | train_mse={epoch_mse:.4f} | val_bler={val_bler:.4f}")
            hyp_str = "  ".join(f"h{j}={epoch_mse_per_hyp[j]:.4f}" for j in range(4))
            print(f"  [train_mse/hyp] {hyp_str}")

            if val_bler < best_val_bler:
                best_val_bler = val_bler
                epochs_no_improve = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                ckpt_path = checkpoint_dir / f"best_model_seed{seed}.pt"
                torch.save(
                    {
                        'model_state': model.state_dict(),
                        'config': config,
                        'epoch': epoch + 1,
                        'val_bler': val_bler,
                        'seed': seed,
                    },
                    ckpt_path,
                )
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= config['patience']:
                print(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {config['patience']} validations)"
                )
                break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save training history
    log_dir = Path(config['checkpoint_dir']).parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        log_dir / f"history_seed{seed}.npz",
        train_mse=np.array(history['train_mse']),
        train_mse_h0=np.array(history['train_mse_h0']),
        train_mse_h1=np.array(history['train_mse_h1']),
        train_mse_h2=np.array(history['train_mse_h2']),
        train_mse_h3=np.array(history['train_mse_h3']),
        val_bler=np.array(history['val_bler']),
        val_epoch=np.array(history['val_epoch']),
    )
    print(f"History saved to {log_dir / f'history_seed{seed}.npz'}")

    return model, history


# ─────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str | Path) -> tuple[NeuralBranchMetric, dict]:
    """
    Load a trained NeuralBranchMetric from a checkpoint file.

    Returns:
        (model, checkpoint_dict)  — model is on CPU in eval mode
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    cfg = ckpt['config']
    model = NeuralBranchMetric(
        hidden_size=cfg['hidden_size'],
        num_layers=cfg['num_layers'],
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt


# ─────────────────────────────────────────────
# Smoke test / CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Branch Metric Estimator (N2)")
    parser.add_argument('--train', action='store_true', help='Run full training loop')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--hidden_size', type=int, default=16, help='GRU hidden size per direction')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batches_per_epoch', type=int, default=None, help='Mini-batches per epoch')
    parser.add_argument('--batch_size', type=int, default=None, help='Training batch size')
    parser.add_argument('--val_every', type=int, default=None, help='Run validation every N epochs')
    parser.add_argument('--val_blocks', type=int, default=None, help='Validation blocks per evaluation')
    parser.add_argument(
        '--snr_range',
        type=float,
        nargs=2,
        metavar=('SNR_MIN', 'SNR_MAX'),
        default=None,
        help='Training SNR range in dB',
    )
    parser.add_argument(
        '--inr_range',
        type=float,
        nargs=2,
        metavar=('INR_MIN', 'INR_MAX'),
        default=None,
        help='Training INR range in dB',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Training device: e.g. 'cpu', 'cuda', or 'mps'. Defaults to auto-detect.",
    )
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False")
        if device == 'mps' and (
            not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available()
        ):
            raise RuntimeError("Requested MPS device, but MPS is not available")

    print("=== NeuralBranchMetric Smoke Test ===")
    print(f"Device: {device}")
    rng = np.random.default_rng(args.seed)

    trellis = load_nasa_k7()
    model = NeuralBranchMetric(hidden_size=args.hidden_size).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Trellis steps: K_INFO info bits + (CONSTRAINT_LEN - 1) tail bits
    T = K_INFO + CONSTRAINT_LEN - 1  # 262

    # 1. Forward pass shape check
    x = torch.randn(1, T, 2, device=device)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == (1, T, 4), f"Expected (1, {T}, 4), got {out.shape}"

    # 2. Generate one real block and compute oracle metrics
    info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
    coded = _encode_fixed_tail(info_bits, trellis)   # always 524 bits
    symbols = bpsk_modulate(coded)
    N = len(symbols)

    received = awgn_channel(symbols, 5.0, 5.0, 16.0, 0.0, rng)
    noise_var = noise_var_from_snr(5.0)
    amplitude = amplitude_from_inr(5.0, noise_var)
    interference = generate_interference(N, amplitude, 16.0, 0.0)

    y_paired = pair_received_signal(received)       # (T, 2)
    i_paired = pair_received_signal(interference)   # (T, 2)
    assert y_paired.shape == (T, 2), f"Paired shape wrong: {y_paired.shape}"

    y_t = torch.tensor(y_paired[np.newaxis], dtype=torch.float32, device=device)   # (1, T, 2)
    i_t = torch.tensor(i_paired[np.newaxis], dtype=torch.float32, device=device)   # (1, T, 2)
    oracle = compute_oracle_metrics(y_t, i_t)
    print(f"Oracle metrics shape: {tuple(oracle.shape)}")
    assert oracle.shape == (1, T, 4)

    # 3. Viterbi with random branch metrics
    index_table = build_branch_output_index(trellis)
    bm_np = out.squeeze(0).detach().cpu().numpy()  # (T, 4)
    decoded = viterbi_neural_bm(bm_np, trellis, index_table)
    print(f"Decoded bits shape:   {decoded.shape}")
    assert decoded.shape == (K_INFO,)

    # 4. make_decoder_n2 interface
    decode_fn = make_decoder_n2(model, device, trellis, index_table)
    result = decode_fn(received, 16.0, 0.0, 5.0, 5.0)
    assert result.shape == (K_INFO,), f"decode_fn output shape wrong: {result.shape}"
    print(f"make_decoder_n2 output shape: {result.shape}")

    print("\nAll smoke tests PASSED")

    if args.train:
        print("\n=== Starting Training ===")
        cfg = {
            **DEFAULT_CONFIG,
            'hidden_size': args.hidden_size,
            'device': device,
        }
        if args.num_epochs is not None:
            cfg['num_epochs'] = args.num_epochs
        if args.batches_per_epoch is not None:
            cfg['batches_per_epoch'] = args.batches_per_epoch
        if args.batch_size is not None:
            cfg['batch_size'] = args.batch_size
        if args.val_every is not None:
            cfg['val_every'] = args.val_every
        if args.val_blocks is not None:
            cfg['val_blocks'] = args.val_blocks
        if args.snr_range is not None:
            cfg['snr_range'] = tuple(args.snr_range)
        if args.inr_range is not None:
            cfg['inr_range'] = tuple(args.inr_range)
        trained, history = train_neural_bm(cfg, seed=args.seed)
        n_p = sum(p.numel() for p in trained.parameters())
        print(f"Training complete. Model params: {n_p:,}")
