from __future__ import annotations

import numpy as np

from stages.s09_regularization.starter.regularization import (
    elastic_net_gradient,
    elastic_net_penalty,
    lasso_penalty,
    lasso_subgradient,
    ridge_gradient,
    ridge_penalty,
)


class TestRidgePenalty:
    def test_basic(self) -> None:
        weights = np.array([1.0, 2.0, 3.0])
        penalty = ridge_penalty(weights, alpha=0.1)
        expected = 0.05 * (1 + 4 + 9)
        np.testing.assert_allclose(penalty, expected)

    def test_zero_weights(self) -> None:
        weights = np.zeros(5)
        penalty = ridge_penalty(weights, alpha=1.0)
        assert penalty == 0.0

    def test_zero_alpha(self) -> None:
        weights = np.array([1.0, 2.0, 3.0])
        penalty = ridge_penalty(weights, alpha=0.0)
        assert penalty == 0.0

    def test_single_weight(self) -> None:
        weights = np.array([3.0])
        penalty = ridge_penalty(weights, alpha=2.0)
        expected = 1.0 * 9
        np.testing.assert_allclose(penalty, expected)

    def test_negative_weights(self) -> None:
        weights = np.array([-1.0, -2.0])
        penalty = ridge_penalty(weights, alpha=1.0)
        expected = 0.5 * (1 + 4)
        np.testing.assert_allclose(penalty, expected)


class TestRidgeGradient:
    def test_basic(self) -> None:
        weights = np.array([1.0, 2.0, 3.0])
        gradient = ridge_gradient(weights, alpha=0.1)
        expected = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(gradient, expected)

    def test_zero_weights(self) -> None:
        weights = np.zeros(5)
        gradient = ridge_gradient(weights, alpha=1.0)
        np.testing.assert_array_equal(gradient, np.zeros(5))

    def test_zero_alpha(self) -> None:
        weights = np.array([1.0, 2.0, 3.0])
        gradient = ridge_gradient(weights, alpha=0.0)
        np.testing.assert_array_equal(gradient, np.zeros(3))

    def test_shape_preserved(self) -> None:
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        gradient = ridge_gradient(weights, alpha=0.5)
        assert gradient.shape == weights.shape


class TestLassoPenalty:
    def test_basic(self) -> None:
        weights = np.array([1.0, -2.0, 3.0])
        penalty = lasso_penalty(weights, alpha=0.1)
        expected = 0.1 * (1 + 2 + 3)
        np.testing.assert_allclose(penalty, expected)

    def test_zero_weights(self) -> None:
        weights = np.zeros(5)
        penalty = lasso_penalty(weights, alpha=1.0)
        assert penalty == 0.0

    def test_zero_alpha(self) -> None:
        weights = np.array([1.0, 2.0, 3.0])
        penalty = lasso_penalty(weights, alpha=0.0)
        assert penalty == 0.0

    def test_all_negative(self) -> None:
        weights = np.array([-1.0, -2.0, -3.0])
        penalty = lasso_penalty(weights, alpha=1.0)
        expected = 1 + 2 + 3
        np.testing.assert_allclose(penalty, expected)


class TestLassoSubgradient:
    def test_positive_weights(self) -> None:
        weights = np.array([1.0, 2.0, 3.0])
        subgrad = lasso_subgradient(weights, alpha=0.1)
        expected = np.array([0.1, 0.1, 0.1])
        np.testing.assert_allclose(subgrad, expected)

    def test_negative_weights(self) -> None:
        weights = np.array([-1.0, -2.0, -3.0])
        subgrad = lasso_subgradient(weights, alpha=0.1)
        expected = np.array([-0.1, -0.1, -0.1])
        np.testing.assert_allclose(subgrad, expected)

    def test_mixed_weights(self) -> None:
        weights = np.array([1.0, -2.0, 0.0])
        subgrad = lasso_subgradient(weights, alpha=0.1)
        expected = np.array([0.1, -0.1, 0.0])
        np.testing.assert_allclose(subgrad, expected)

    def test_zero_at_zero(self) -> None:
        weights = np.array([0.0, 0.0, 0.0])
        subgrad = lasso_subgradient(weights, alpha=1.0)
        np.testing.assert_array_equal(subgrad, np.zeros(3))


class TestElasticNetPenalty:
    def test_pure_l1(self) -> None:
        weights = np.array([1.0, 2.0])
        penalty = elastic_net_penalty(weights, alpha=1.0, l1_ratio=1.0)
        expected = lasso_penalty(weights, alpha=1.0)
        np.testing.assert_allclose(penalty, expected)

    def test_pure_l2(self) -> None:
        weights = np.array([1.0, 2.0])
        penalty = elastic_net_penalty(weights, alpha=1.0, l1_ratio=0.0)
        expected = ridge_penalty(weights, alpha=1.0)
        np.testing.assert_allclose(penalty, expected)

    def test_mixed(self) -> None:
        weights = np.array([1.0, 2.0])
        penalty = elastic_net_penalty(weights, alpha=1.0, l1_ratio=0.5)
        l1 = 0.5 * (1 + 2)
        l2 = 0.25 * (1 + 4)
        expected = l1 + l2
        np.testing.assert_allclose(penalty, expected)

    def test_zero_alpha(self) -> None:
        weights = np.array([1.0, 2.0, 3.0])
        penalty = elastic_net_penalty(weights, alpha=0.0, l1_ratio=0.5)
        assert penalty == 0.0


class TestElasticNetGradient:
    def test_pure_l1(self) -> None:
        weights = np.array([1.0, -2.0])
        grad = elastic_net_gradient(weights, alpha=1.0, l1_ratio=1.0)
        expected = lasso_subgradient(weights, alpha=1.0)
        np.testing.assert_allclose(grad, expected)

    def test_pure_l2(self) -> None:
        weights = np.array([1.0, -2.0])
        grad = elastic_net_gradient(weights, alpha=1.0, l1_ratio=0.0)
        expected = ridge_gradient(weights, alpha=1.0)
        np.testing.assert_allclose(grad, expected)

    def test_mixed(self) -> None:
        weights = np.array([1.0, -2.0])
        grad = elastic_net_gradient(weights, alpha=1.0, l1_ratio=0.5)
        l1_grad = np.array([0.5, -0.5])
        l2_grad = np.array([0.5, -1.0])
        expected = l1_grad + l2_grad
        np.testing.assert_allclose(grad, expected)

    def test_shape_preserved(self) -> None:
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        grad = elastic_net_gradient(weights, alpha=0.5, l1_ratio=0.5)
        assert grad.shape == weights.shape
