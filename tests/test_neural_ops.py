"""
test_neural_ops.py — Verify BiGRU decoder compute budget compliance

Tests that the neural decoder stays within the 2.1M operation budget
and produces correct output shapes.
"""

import pytest
import torch
import numpy as np

from neural_decoder import BiRNNDecoder, K_INFO
from compute_cost import (
    OP_BUDGET, count_flops_neural, count_flops_birnn_analytical,
    assert_within_budget,
)


# Trellis step count: 256 info + 6 tail = 262
T_STEPS = 262
INPUT_DIM = 2


@pytest.fixture
def bigru_model():
    return BiRNNDecoder(hidden_size=24, input_dim=INPUT_DIM, cell_type="GRU",
                        bidirectional=True)


@pytest.fixture
def sample_input():
    return torch.randn(1, T_STEPS, INPUT_DIM)


class TestOutputShape:
    """Verify model produces correct output dimensions."""

    def test_single_sample(self, bigru_model, sample_input):
        bigru_model.eval()
        with torch.no_grad():
            out = bigru_model(sample_input)
        assert out.shape == (1, K_INFO), f"Expected (1, {K_INFO}), got {out.shape}"

    def test_batch(self, bigru_model):
        bigru_model.eval()
        x = torch.randn(16, T_STEPS, INPUT_DIM)
        with torch.no_grad():
            out = bigru_model(x)
        assert out.shape == (16, K_INFO), f"Expected (16, {K_INFO}), got {out.shape}"

    def test_output_is_logits(self, bigru_model, sample_input):
        """Output should be raw logits (not bounded to [0,1])."""
        bigru_model.eval()
        with torch.no_grad():
            out = bigru_model(sample_input)
        # Logits can be any real number; if all are in [0,1], sigmoid was applied
        # This is a probabilistic check — with random weights, some should be outside [0,1]
        assert out.min() < 0.0 or out.max() > 1.0, \
            "Output looks like probabilities, not logits"


class TestComputeBudget:
    """Verify BiGRU stays within 2.1M operation budget."""

    def test_analytical_bigru_h24(self):
        flops = count_flops_birnn_analytical(
            hidden_size=24, input_dim=INPUT_DIM, seq_len=T_STEPS,
            cell_type="GRU", bidirectional=True,
        )
        assert flops <= OP_BUDGET, \
            f"BiGRU h=24 analytical: {flops:,} > {OP_BUDGET:,}"
        print(f"BiGRU h=24 analytical: {flops:,} ops ({flops/OP_BUDGET*100:.1f}%)")

    def test_thop_bigru_h24(self, bigru_model, sample_input):
        """Test with thop if available, otherwise skip."""
        try:
            import thop
        except ImportError:
            pytest.skip("thop not installed")

        flops = count_flops_neural(bigru_model, sample_input)
        # thop may count differently; check within 2x of budget as sanity
        assert flops <= OP_BUDGET * 2, \
            f"BiGRU h=24 thop: {flops:,} > {OP_BUDGET * 2:,}"
        print(f"BiGRU h=24 thop: {flops:,} ops")

    def test_analytical_budget_assertion(self):
        """assert_within_budget should not raise for BiGRU h=24."""
        flops = count_flops_birnn_analytical(
            hidden_size=24, input_dim=INPUT_DIM, seq_len=T_STEPS,
            cell_type="GRU", bidirectional=True,
        )
        # Should not raise
        assert_within_budget("N1_gru_e2e", flops)


class TestParameterCount:
    """Verify parameter count matches expectations."""

    def test_bigru_param_count(self, bigru_model):
        n_params = sum(p.numel() for p in bigru_model.parameters())
        # BiGRU h=24, d=2: per direction 3 gates * (24*(24+2) + 24) = 3*(624+24) = 1944
        # Two directions = 3888, output linear = 48+1 = 49, total ~ 3937
        assert 3000 < n_params < 6000, \
            f"Parameter count {n_params} outside expected range [3000, 6000]"
        print(f"BiGRU h=24 parameters: {n_params}")
