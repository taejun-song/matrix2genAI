from __future__ import annotations

import numpy as np

from stages.s07_logistic_regression.starter.losses import (
    binary_cross_entropy,
    binary_cross_entropy_gradient,
    categorical_cross_entropy,
    categorical_cross_entropy_gradient,
)


class TestBinaryCrossEntropy:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_pred_proba = np.array([1.0, 0.0, 1.0, 0.0])

        loss = binary_cross_entropy(y_true, y_pred_proba)

        assert loss < 1e-10

    def test_worst_predictions(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_pred_proba = np.array([0.0, 1.0, 0.0, 1.0])

        loss = binary_cross_entropy(y_true, y_pred_proba)

        assert loss > 10

    def test_random_predictions(self) -> None:
        y_true = np.array([1, 0, 1])
        y_pred_proba = np.array([0.9, 0.1, 0.8])

        loss = binary_cross_entropy(y_true, y_pred_proba)

        expected = -(np.log(0.9) + np.log(0.9) + np.log(0.8)) / 3
        np.testing.assert_allclose(loss, expected, rtol=1e-5)

    def test_numerical_stability(self) -> None:
        y_true = np.array([1, 0])
        y_pred_proba = np.array([1.0, 0.0])

        loss = binary_cross_entropy(y_true, y_pred_proba)

        assert np.isfinite(loss)

    def test_uniform_predictions(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_pred_proba = np.array([0.5, 0.5, 0.5, 0.5])

        loss = binary_cross_entropy(y_true, y_pred_proba)

        expected = -np.log(0.5)
        np.testing.assert_allclose(loss, expected)


class TestCategoricalCrossEntropy:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred_proba = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        loss = categorical_cross_entropy(y_true, y_pred_proba)

        assert loss < 1e-10

    def test_worst_predictions(self) -> None:
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred_proba = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])

        loss = categorical_cross_entropy(y_true, y_pred_proba)

        assert loss > 10

    def test_random_predictions(self) -> None:
        y_true = np.array([[1, 0], [0, 1]])
        y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8]])

        loss = categorical_cross_entropy(y_true, y_pred_proba)

        expected = -(np.log(0.9) + np.log(0.8)) / 2
        np.testing.assert_allclose(loss, expected, rtol=1e-5)

    def test_numerical_stability(self) -> None:
        y_true = np.array([[1, 0, 0]])
        y_pred_proba = np.array([[1.0, 0.0, 0.0]])

        loss = categorical_cross_entropy(y_true, y_pred_proba)

        assert np.isfinite(loss)


class TestBinaryCrossEntropyGradient:
    def test_zero_gradient(self) -> None:
        X = np.array([[1, 2], [3, 4]])
        y_true = np.array([1, 0])
        y_pred_proba = np.array([1.0, 0.0])

        grad_w, grad_b = binary_cross_entropy_gradient(X, y_true, y_pred_proba)

        np.testing.assert_allclose(grad_w, [0, 0], atol=1e-10)
        np.testing.assert_allclose(grad_b, 0.0, atol=1e-10)

    def test_simple_gradient(self) -> None:
        X = np.array([[1, 2], [3, 4]])
        y_true = np.array([1, 0])
        y_pred_proba = np.array([0.9, 0.2])

        grad_w, grad_b = binary_cross_entropy_gradient(X, y_true, y_pred_proba)

        errors = y_pred_proba - y_true
        expected_grad_w = (1 / 2) * X.T @ errors
        expected_grad_b = (1 / 2) * np.sum(errors)

        np.testing.assert_allclose(grad_w, expected_grad_w)
        np.testing.assert_allclose(grad_b, expected_grad_b)

    def test_gradient_shape(self) -> None:
        X = np.array([[1, 2, 3], [4, 5, 6]])
        y_true = np.array([1, 0])
        y_pred_proba = np.array([0.7, 0.3])

        grad_w, grad_b = binary_cross_entropy_gradient(X, y_true, y_pred_proba)

        assert grad_w.shape == (3,)
        assert isinstance(grad_b, (float, np.floating))

    def test_gradient_direction(self) -> None:
        X = np.array([[1, 1]])
        y_true = np.array([1])
        y_pred_proba = np.array([0.3])

        grad_w, grad_b = binary_cross_entropy_gradient(X, y_true, y_pred_proba)

        assert grad_b < 0


class TestCategoricalCrossEntropyGradient:
    def test_zero_gradient(self) -> None:
        X = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1, 0], [0, 1]])
        y_pred_proba = np.array([[1.0, 0.0], [0.0, 1.0]])

        grad_w, grad_b = categorical_cross_entropy_gradient(X, y_true, y_pred_proba)

        np.testing.assert_allclose(grad_w, np.zeros((2, 2)), atol=1e-10)
        np.testing.assert_allclose(grad_b, np.zeros(2), atol=1e-10)

    def test_simple_gradient(self) -> None:
        X = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1, 0], [0, 1]])
        y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8]])

        grad_w, grad_b = categorical_cross_entropy_gradient(X, y_true, y_pred_proba)

        errors = y_pred_proba - y_true
        expected_grad_w = (1 / 2) * X.T @ errors
        expected_grad_b = (1 / 2) * np.sum(errors, axis=0)

        np.testing.assert_allclose(grad_w, expected_grad_w)
        np.testing.assert_allclose(grad_b, expected_grad_b)

    def test_gradient_shape(self) -> None:
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred_proba = np.array(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        )

        grad_w, grad_b = categorical_cross_entropy_gradient(X, y_true, y_pred_proba)

        assert grad_w.shape == (2, 3)
        assert grad_b.shape == (3,)

    def test_three_classes(self) -> None:
        X = np.array([[1, 1]])
        y_true = np.array([[1, 0, 0]])
        y_pred_proba = np.array([[0.5, 0.3, 0.2]])

        grad_w, grad_b = categorical_cross_entropy_gradient(X, y_true, y_pred_proba)

        assert grad_w.shape == (2, 3)
        assert grad_b.shape == (3,)
