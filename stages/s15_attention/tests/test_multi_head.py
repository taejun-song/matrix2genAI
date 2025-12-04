from __future__ import annotations

import numpy as np

from stages.s15_attention.starter.multi_head import (
    MultiHeadAttention,
    merge_heads,
    multi_head_attention_forward,
    split_heads,
)


class TestSplitHeads:
    def test_output_shape(self) -> None:
        x = np.random.randn(2, 10, 512)
        split = split_heads(x, num_heads=8)
        assert split.shape == (2, 8, 10, 64)

    def test_preserves_values(self) -> None:
        x = np.arange(24).reshape(1, 2, 12)
        split = split_heads(x, num_heads=3)
        assert split.shape == (1, 3, 2, 4)

    def test_single_head(self) -> None:
        x = np.random.randn(2, 5, 64)
        split = split_heads(x, num_heads=1)
        assert split.shape == (2, 1, 5, 64)


class TestMergeHeads:
    def test_output_shape(self) -> None:
        x = np.random.randn(2, 8, 10, 64)
        merged = merge_heads(x)
        assert merged.shape == (2, 10, 512)

    def test_inverse_of_split(self) -> None:
        x = np.random.randn(2, 10, 512)
        split = split_heads(x, num_heads=8)
        merged = merge_heads(split)
        np.testing.assert_array_equal(x, merged)

    def test_single_head(self) -> None:
        x = np.random.randn(2, 1, 5, 64)
        merged = merge_heads(x)
        assert merged.shape == (2, 5, 64)


class TestMultiHeadAttentionForward:
    def test_output_shape(self) -> None:
        np.random.seed(42)
        batch, seq, d_model = 2, 10, 64
        num_heads = 4
        Q = np.random.randn(batch, seq, d_model)
        K = np.random.randn(batch, seq, d_model)
        V = np.random.randn(batch, seq, d_model)
        W_Q = np.random.randn(d_model, d_model)
        W_K = np.random.randn(d_model, d_model)
        W_V = np.random.randn(d_model, d_model)
        W_O = np.random.randn(d_model, d_model)
        output, _ = multi_head_attention_forward(Q, K, V, W_Q, W_K, W_V, W_O, num_heads)
        assert output.shape == (batch, seq, d_model)

    def test_different_kv_length(self) -> None:
        np.random.seed(42)
        batch, seq_q, seq_k, d_model = 2, 5, 10, 64
        num_heads = 4
        Q = np.random.randn(batch, seq_q, d_model)
        K = np.random.randn(batch, seq_k, d_model)
        V = np.random.randn(batch, seq_k, d_model)
        W_Q = np.random.randn(d_model, d_model)
        W_K = np.random.randn(d_model, d_model)
        W_V = np.random.randn(d_model, d_model)
        W_O = np.random.randn(d_model, d_model)
        output, _ = multi_head_attention_forward(Q, K, V, W_Q, W_K, W_V, W_O, num_heads)
        assert output.shape == (batch, seq_q, d_model)

    def test_cache_contents(self) -> None:
        np.random.seed(42)
        batch, seq, d_model = 2, 5, 32
        num_heads = 4
        Q = np.random.randn(batch, seq, d_model)
        K = np.random.randn(batch, seq, d_model)
        V = np.random.randn(batch, seq, d_model)
        W_Q = np.random.randn(d_model, d_model)
        W_K = np.random.randn(d_model, d_model)
        W_V = np.random.randn(d_model, d_model)
        W_O = np.random.randn(d_model, d_model)
        _, cache = multi_head_attention_forward(Q, K, V, W_Q, W_K, W_V, W_O, num_heads)
        assert 'Q_heads' in cache
        assert 'weights' in cache
        assert cache['weights'].shape == (batch, num_heads, seq, seq)


class TestMultiHeadAttentionClass:
    def test_initialization(self) -> None:
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        params = mha.get_params()
        assert params['W_Q'].shape == (512, 512)
        assert params['W_K'].shape == (512, 512)
        assert params['W_V'].shape == (512, 512)
        assert params['W_O'].shape == (512, 512)

    def test_forward(self) -> None:
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        x = np.random.randn(2, 10, 64)
        output = mha.forward(x, x, x)
        assert output.shape == (2, 10, 64)

    def test_self_attention(self) -> None:
        mha = MultiHeadAttention(d_model=32, num_heads=2)
        x = np.random.randn(1, 5, 32)
        output = mha.forward(x, x, x)
        assert output.shape == x.shape

    def test_with_mask(self) -> None:
        mha = MultiHeadAttention(d_model=32, num_heads=2)
        x = np.random.randn(1, 4, 32)
        mask = np.tril(np.ones((4, 4), dtype=bool))
        output = mha.forward(x, x, x, mask)
        assert output.shape == x.shape
