from __future__ import annotations

import numpy as np

from stages.s15_attention.starter.attention_core import (
    apply_attention_mask,
    attention_weights,
    compute_attention_scores,
    scaled_dot_product_attention,
)


class TestComputeAttentionScores:
    def test_output_shape(self) -> None:
        Q = np.random.randn(2, 4, 64)
        K = np.random.randn(2, 6, 64)
        scores = compute_attention_scores(Q, K)
        assert scores.shape == (2, 4, 6)

    def test_scaling(self) -> None:
        np.random.seed(42)
        Q = np.random.randn(1, 1, 64)
        K = np.random.randn(1, 1, 64)
        scaled = compute_attention_scores(Q, K, scale=True)
        unscaled = compute_attention_scores(Q, K, scale=False)
        np.testing.assert_allclose(scaled * np.sqrt(64), unscaled)

    def test_self_attention_diagonal(self) -> None:
        x = np.eye(4).reshape(1, 4, 4)
        scores = compute_attention_scores(x, x, scale=False)
        assert scores[0, 0, 0] == 1.0
        assert scores[0, 0, 1] == 0.0

    def test_batched(self) -> None:
        Q = np.random.randn(8, 10, 32)
        K = np.random.randn(8, 10, 32)
        scores = compute_attention_scores(Q, K)
        assert scores.shape == (8, 10, 10)


class TestApplyAttentionMask:
    def test_masking(self) -> None:
        scores = np.array([[1.0, 2.0, 3.0]])
        mask = np.array([[True, True, False]])
        masked = apply_attention_mask(scores, mask)
        assert masked[0, 2] < -1e8
        assert masked[0, 0] == 1.0

    def test_no_mask(self) -> None:
        scores = np.random.randn(2, 4, 4)
        mask = np.ones((2, 4, 4), dtype=bool)
        masked = apply_attention_mask(scores, mask)
        np.testing.assert_array_equal(scores, masked)

    def test_causal_mask(self) -> None:
        scores = np.ones((1, 4, 4))
        mask = np.tril(np.ones((4, 4), dtype=bool))
        masked = apply_attention_mask(scores, mask)
        assert masked[0, 0, 1] < -1e8
        assert masked[0, 1, 1] == 1.0


class TestAttentionWeights:
    def test_sums_to_one(self) -> None:
        scores = np.random.randn(2, 4, 6)
        weights = attention_weights(scores)
        sums = weights.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    def test_all_positive(self) -> None:
        scores = np.random.randn(2, 4, 6)
        weights = attention_weights(scores)
        assert np.all(weights >= 0)

    def test_numerical_stability(self) -> None:
        scores = np.array([[1000.0, 1000.0, 1000.0]])
        weights = attention_weights(scores)
        assert np.all(np.isfinite(weights))
        np.testing.assert_allclose(weights.sum(), 1.0)


class TestScaledDotProductAttention:
    def test_output_shape(self) -> None:
        Q = np.random.randn(2, 4, 64)
        K = np.random.randn(2, 6, 64)
        V = np.random.randn(2, 6, 128)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (2, 4, 128)
        assert weights.shape == (2, 4, 6)

    def test_weights_sum_to_one(self) -> None:
        Q = np.random.randn(2, 4, 64)
        K = np.random.randn(2, 6, 64)
        V = np.random.randn(2, 6, 64)
        _, weights = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, rtol=1e-5)

    def test_with_mask(self) -> None:
        Q = np.ones((1, 3, 4))
        K = np.ones((1, 3, 4))
        V = np.arange(9).reshape(1, 3, 3).astype(float)
        mask = np.array([[[True, False, False],
                          [True, True, False],
                          [True, True, True]]])
        output, weights = scaled_dot_product_attention(Q, K, V, mask)
        np.testing.assert_allclose(weights[0, 0, 0], 1.0, rtol=1e-5)

    def test_identical_qk_attends_uniformly(self) -> None:
        Q = np.ones((1, 4, 8))
        K = np.ones((1, 4, 8))
        V = np.random.randn(1, 4, 8)
        _, weights = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_allclose(weights, 0.25, rtol=1e-5)
