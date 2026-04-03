"""
test_neural_ops.py — Verify BiGRU decoder output shapes and parameter count

Per CLAUDE.md: h=32/dir, input_dim=1, N=524 time steps, ~6593 params.
Note: h=32 gives ~3.3M ops (exceeds 2.1M budget). Budget trimming is
a separate step — this test verifies the architecture is correct.
"""

import pytest
import torch

from neural_decoder import BiRNNDecoder, K_INFO
from compute_cost import count_flops_birnn_analytical


N_SYM = 524
INPUT_DIM = 1


@pytest.fixture
def bigru_model():
    return BiRNNDecoder(hidden_size=32, input_dim=INPUT_DIM, cell_type="GRU",
                        bidirectional=True)


@pytest.fixture
def sample_input():
    return torch.randn(1, N_SYM, INPUT_DIM)


class TestOutputShape:

    def test_single_sample(self, bigru_model, sample_input):
        bigru_model.eval()
        with torch.no_grad():
            out = bigru_model(sample_input)
        assert out.shape == (1, K_INFO), f"Expected (1, {K_INFO}), got {out.shape}"

    def test_batch(self, bigru_model):
        bigru_model.eval()
        x = torch.randn(16, N_SYM, INPUT_DIM)
        with torch.no_grad():
            out = bigru_model(x)
        assert out.shape == (16, K_INFO)

    def test_output_is_logits(self, bigru_model, sample_input):
        bigru_model.eval()
        with torch.no_grad():
            out = bigru_model(sample_input)
        assert out.min() < 0.0 or out.max() > 1.0, \
            "Output looks like probabilities, not logits"


class TestParameterCount:

    def test_bigru_h32_param_count(self, bigru_model):
        """Per CLAUDE.md: ~6593 params for h=32, input=1."""
        n_params = sum(p.numel() for p in bigru_model.parameters())
        # GRU per dir: 3 gates * (h*(h+d) + h) = 3*(32*33+32) = 3264
        # Two dirs: 6528, output: 64+1=65, total: 6593
        assert 6000 < n_params < 7500, \
            f"Parameter count {n_params} outside expected range"
        print(f"BiGRU h=32 parameters: {n_params}")


class TestAnalyticalOps:

    def test_bigru_h32_ops(self):
        """h=32 at N=524, d=1 gives ~6.6M ops. Train first, trim later."""
        flops = count_flops_birnn_analytical(
            hidden_size=32, input_dim=INPUT_DIM, seq_len=N_SYM,
            cell_type="GRU", bidirectional=True,
        )
        assert 6_000_000 < flops < 7_500_000, \
            f"BiGRU h=32 ops: {flops:,}"
        print(f"BiGRU h=32 analytical ops: {flops:,}")

    def test_find_budget_hidden_size(self):
        """Find the hidden size that fits within 2.1M ops at N=524, d=1."""
        for h in range(10, 33):
            flops = count_flops_birnn_analytical(
                hidden_size=h, input_dim=INPUT_DIM, seq_len=N_SYM,
                cell_type="GRU", bidirectional=True,
            )
            if flops <= 2_100_000:
                best_h = h
        print(f"Max h within 2.1M budget: {best_h} "
              f"({count_flops_birnn_analytical(best_h, INPUT_DIM, N_SYM, 'GRU', True):,} ops)")
