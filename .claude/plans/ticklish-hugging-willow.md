# Phase 2a: GRU End-to-End Neural Decoder (N1)

## Context

Phase 1 (classical baselines B1/B2/B5) is complete. Phase 2a implements a bidirectional RNN decoder that takes raw received signals and outputs decoded info bits, learning to handle sinusoidal interference without explicit interference knowledge. The goal is to beat mismatched Viterbi (B1) by ≥0.5 dB at BLER=10⁻³.

## Critical Finding: Sequence Length is 524, Not 512

The `N_CODED=512` constant is misleading. Actual pipeline:
- `trellis.encode(info_bits)` → 2*(256+6) = **524 coded bits** (256 info + 6 tail, rate 1/2)
- `bpsk_modulate(coded)` → **524 BPSK symbols**
- Channel → **524 received samples**
- Viterbi reshapes to **(262, 2)** — 262 trellis time steps
- Returns `decoded[:256]` — first 256 info bits

The neural decoder must handle T=262 time steps with input_dim=2.

---

## Step 1: Create `neural_decoder.py`

### Architecture: `BiRNNDecoder(nn.Module)`

| Parameter | BiGRU (primary) | BiLSTM | BiRNN (vanilla) |
|-----------|----------------|--------|-----------------|
| hidden_size/dir | 24 | 20 | 40 |
| input_dim | 2 | 2 | 2 |
| bidirectional | Yes | Yes | Yes |
| Output dim | 48 (24×2) | 40 (20×2) | 80 (40×2) |
| Output layer | Linear(out_dim, 1) | same | same |
| Approx ops | ~2.0M | ~1.9M | ~1.8M |

**Hidden sizes must be verified with thop and adjusted to stay ≤ 2.1M ops.**

GRU op formula: `total = 2 (bidir) × T × 6 × h × (h + d)` where T=262, d=2.
- h=24: `2 × 262 × 6 × 24 × 26 = 1,961,856` + linear ~25K ≈ **1.99M** ✓

Forward pass:
```
input: (batch, 262, 2)
→ BiRNN → (batch, 262, 2*hidden)
→ Linear(2*hidden, 1) → (batch, 262, 1) → squeeze → (batch, 262)
→ return logits[:, :256]  (discard 6 tail positions)
```

Use `BCEWithLogitsLoss` during training (numerically stable). Apply sigmoid only at inference.

### Decode wrapper: `make_decoder_n1(model, device)`

Returns `decode_fn(received, period, phase, snr_db, inr_db) -> ndarray(256,)` compatible with `eval.py`.
- Reshape received (524,) → (1, 262, 2)
- Forward pass → sigmoid → threshold 0.5 → return first 256 bits as int8

### Training: `train_model(config) -> model`

**Data generation strategy**: Pre-encode a pool of 50,000 (info_bits, coded_symbols) pairs to avoid the Python-loop encoding bottleneck. During training, sample from pool and apply fresh channel noise/interference per batch.

- Batch size: 128
- Batches/epoch: 1,000 (128K blocks/epoch)
- Epochs: 100 (total: 12.8M blocks)
- Loss: BCE on 256 info bit positions
- Optimizer: Adam, lr=1e-3
- Scheduler: ReduceLROnPlateau(patience=10, factor=0.1)
- Gradient clipping: max_norm=1.0
- Validation: 10,000 blocks at SNR=5dB, INR=5dB every epoch
- Early stopping: patience=20
- Checkpoints saved to `results/phase2a/checkpoints/`

Training channel ranges:
- SNR ∈ U(0, 10) dB
- INR ∈ U(-5, 10) dB  (cap at 10, not 15, to test generalization)
- P ∈ U(8, 32), φ ∈ U(0, 2π)

### Smoke test in `__main__`

Instantiate model, run forward pass on random input, verify output shape.

---

## Step 2: Update `compute_cost.py`

Add analytical op counters as cross-check alongside thop:
- `count_flops_bigru_analytical(hidden, input_dim, seq_len)` → `2 * T * 6 * h * (h + d)`
- `count_flops_bilstm_analytical(hidden, input_dim, seq_len)` → `2 * T * 8 * h * (h + d)`
- `count_flops_birnn_analytical(hidden, input_dim, seq_len)` → `2 * T * 2 * h * (h + d)`

---

## Step 3: Create tests

### `tests/test_neural_ops.py`
- Instantiate BiGRU(h=24), BiLSTM(h=20), BiRNN(h=40)
- Verify each ≤ 2.1M ops via `count_flops_neural()` or analytical formula
- Verify output shape is (batch, 256) for input (batch, 262, 2)

### `tests/test_neural_overfit.py`
- Generate 1 fixed batch of 32 blocks (SNR=10dB, INR=0dB)
- Train BiGRU for 200 steps on this single batch
- Assert loss < 0.1 and bit accuracy > 95%

### `tests/test_neural_vs_b1.py` (`@pytest.mark.slow`)
- Load pre-trained checkpoint or train small model (fewer epochs)
- Compare N1 vs B1 BLER at SNR=5dB, INR=5dB
- Assert N1_BLER < B1_BLER (basic superiority)

---

## Step 4: Train all 3 variants

1. BiGRU h=24 (primary)
2. BiLSTM h=20 (comparison)
3. BiRNN h=40 (sanity check — expected to underperform)

Save to `results/phase2a/checkpoints/best_{gru,lstm,rnn}.pt` and training logs to `results/phase2a/logs/`.

---

## Step 5: Evaluate and generate figures

Run using `eval.py` infrastructure:
1. **BLER vs SNR** sweep at INR=5dB, SNR 0-10 dB — methods: B1, B2, B5, N1(GRU), N1(LSTM), N1(RNN)
2. **BLER vs INR** sweep at SNR=5dB, INR -5 to 15 dB
3. **Generalization tests**: INR=12,15 dB (outside training range); P=4,48 (outside [8,32])
4. **Compute table**: FLOPs, latency, BLER@5dB for all methods
5. Use `db_gain()` from `plot_utils.py` to measure N1 vs B1 improvement at BLER=10⁻³

Figures saved to `results/phase2a/figures/`:
- `phase2a_bler_vs_snr.pdf`
- `phase2a_bler_vs_inr.pdf`
- `training_loss.pdf`

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `neural_decoder.py` | **Create** — BiRNNDecoder, train_model, make_decoder_n1 |
| `compute_cost.py` | **Modify** — add analytical RNN op counters |
| `tests/test_neural_ops.py` | **Create** — op budget verification |
| `tests/test_neural_overfit.py` | **Create** — single-batch overfit test |
| `tests/test_neural_vs_b1.py` | **Create** — N1 > B1 integration test |
| `eval.py` | **Modify** — add Phase 2a runner (sweep with neural methods) |

### Existing functions to reuse
- `channel.py`: `bpsk_modulate`, `noise_var_from_snr`, `amplitude_from_inr`, `generate_interference`, `awgn_channel`, `sample_channel_params`
- `trellis.py`: `load_nasa_k7`, `Trellis.encode`
- `eval.py`: `estimate_bler`, `sweep_snr`, `sweep_inr`, `make_encoder`, `make_decoder_b1/b2/b5`
- `compute_cost.py`: `count_flops_neural`, `assert_within_budget`, `make_compute_table`, `measure_latency_ms`
- `plot_utils.py`: `plot_bler_vs_snr`, `plot_bler_vs_inr`, `db_gain`, `METHOD_STYLE['N1_gru_e2e']`

---

## Verification

1. **Op count**: `pytest tests/test_neural_ops.py` — all 3 variants ≤ 2.1M
2. **Learning works**: `pytest tests/test_neural_overfit.py` — overfits single batch
3. **Smoke test**: `python neural_decoder.py` — forward pass succeeds
4. **Training**: Run `train_model()` for all 3 variants, check loss decreases and val BLER improves
5. **N1 > B1**: `pytest tests/test_neural_vs_b1.py` — neural beats mismatched Viterbi
6. **Full eval**: Run SNR/INR sweeps with 100K trials, generate figures, verify ≥0.5 dB gain at BLER=10⁻³
