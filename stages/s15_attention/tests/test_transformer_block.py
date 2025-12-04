from __future__ import annotations

import numpy as np

from stages.s15_attention.starter.transformer_block import (
    TransformerEncoderBlock,
    feed_forward,
    layer_norm,
    stack_encoder_blocks,
)


class TestLayerNorm:
    def test_output_shape(self) -> None:
        x = np.random.randn(2, 10, 64)
        gamma = np.ones(64)
        beta = np.zeros(64)
        output = layer_norm(x, gamma, beta)
        assert output.shape == x.shape

    def test_normalized_stats(self) -> None:
        np.random.seed(42)
        x = np.random.randn(2, 10, 64) * 5 + 3
        gamma = np.ones(64)
        beta = np.zeros(64)
        output = layer_norm(x, gamma, beta)
        means = output.mean(axis=-1)
        stds = output.std(axis=-1)
        np.testing.assert_allclose(means, 0, atol=1e-5)
        np.testing.assert_allclose(stds, 1, atol=1e-2)

    def test_scale_shift(self) -> None:
        x = np.random.randn(1, 5, 4)
        gamma = np.array([2.0, 2.0, 2.0, 2.0])
        beta = np.array([1.0, 1.0, 1.0, 1.0])
        output = layer_norm(x, gamma, beta)
        means = output.mean(axis=-1)
        np.testing.assert_allclose(means, 1.0, atol=1e-5)

    def test_identity_params(self) -> None:
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        gamma = np.ones(4)
        beta = np.zeros(4)
        output = layer_norm(x, gamma, beta)
        assert np.abs(output.mean()) < 1e-5


class TestFeedForward:
    def test_output_shape(self) -> None:
        x = np.random.randn(2, 10, 64)
        W1 = np.random.randn(64, 256)
        b1 = np.zeros(256)
        W2 = np.random.randn(256, 64)
        b2 = np.zeros(64)
        output = feed_forward(x, W1, b1, W2, b2)
        assert output.shape == x.shape

    def test_relu_applied(self) -> None:
        x = np.random.randn(1, 1, 4)
        W1 = np.eye(4)
        b1 = np.array([-10.0, 0.0, 0.0, 0.0])
        W2 = np.eye(4)
        b2 = np.zeros(4)
        output = feed_forward(x, W1, b1, W2, b2)
        assert output[0, 0, 0] >= 0

    def test_expansion_ratio(self) -> None:
        x = np.random.randn(2, 5, 32)
        W1 = np.random.randn(32, 128)
        b1 = np.zeros(128)
        W2 = np.random.randn(128, 32)
        b2 = np.zeros(32)
        output = feed_forward(x, W1, b1, W2, b2)
        assert output.shape == (2, 5, 32)


class TestTransformerEncoderBlock:
    def test_output_shape(self) -> None:
        block = TransformerEncoderBlock(d_model=64, num_heads=4)
        x = np.random.randn(2, 10, 64)
        output = block.forward(x)
        assert output.shape == x.shape

    def test_with_mask(self) -> None:
        block = TransformerEncoderBlock(d_model=32, num_heads=2)
        x = np.random.randn(1, 5, 32)
        mask = np.tril(np.ones((5, 5), dtype=bool))
        output = block.forward(x, mask)
        assert output.shape == x.shape

    def test_default_d_ff(self) -> None:
        block = TransformerEncoderBlock(d_model=64, num_heads=4)
        assert block.d_ff == 256

    def test_custom_d_ff(self) -> None:
        block = TransformerEncoderBlock(d_model=64, num_heads=4, d_ff=128)
        assert block.d_ff == 128


class TestStackEncoderBlocks:
    def test_output_shape(self) -> None:
        blocks = [
            TransformerEncoderBlock(d_model=32, num_heads=2)
            for _ in range(3)
        ]
        x = np.random.randn(2, 10, 32)
        output = stack_encoder_blocks(x, blocks)
        assert output.shape == x.shape

    def test_single_block(self) -> None:
        blocks = [TransformerEncoderBlock(d_model=32, num_heads=2)]
        x = np.random.randn(1, 5, 32)
        output = stack_encoder_blocks(x, blocks)
        assert output.shape == x.shape

    def test_with_mask(self) -> None:
        blocks = [
            TransformerEncoderBlock(d_model=32, num_heads=2)
            for _ in range(2)
        ]
        x = np.random.randn(1, 4, 32)
        mask = np.tril(np.ones((4, 4), dtype=bool))
        output = stack_encoder_blocks(x, blocks, mask)
        assert output.shape == x.shape


class TestTransformerIntegration:
    def test_full_forward_pass(self) -> None:
        np.random.seed(42)
        d_model = 64
        num_heads = 4
        seq_len = 20
        batch_size = 4
        blocks = [
            TransformerEncoderBlock(d_model, num_heads)
            for _ in range(2)
        ]
        x = np.random.randn(batch_size, seq_len, d_model)
        output = stack_encoder_blocks(x, blocks)
        assert output.shape == x.shape
        assert np.all(np.isfinite(output))
