from __future__ import annotations

import numpy as np

from stages.s06_linear_regression.starter.primitives import (
    mse_gradient,
    mse_loss,
    normal_equation,
    polynomial_features,
    predict,
    r2_score,
    standardize,
    train_test_split,
)


class TestPredict:
    def test_simple_prediction(self) -> None:
        X = np.array([[1, 2], [3, 4]])
        weights = np.array([0.5, 1.0])
        bias = 0.1

        y_pred = predict(X, weights, bias)

        expected = np.array([2.6, 5.6])
        np.testing.assert_allclose(y_pred, expected)

    def test_single_feature(self) -> None:
        X = np.array([[1], [2], [3]])
        weights = np.array([2.0])
        bias = 1.0

        y_pred = predict(X, weights, bias)

        expected = np.array([3.0, 5.0, 7.0])
        np.testing.assert_allclose(y_pred, expected)

    def test_zero_bias(self) -> None:
        X = np.array([[1, 1], [2, 2]])
        weights = np.array([1.0, 1.0])
        bias = 0.0

        y_pred = predict(X, weights, bias)

        expected = np.array([2.0, 4.0])
        np.testing.assert_allclose(y_pred, expected)


class TestMSELoss:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        loss = mse_loss(y_true, y_pred)

        np.testing.assert_allclose(loss, 0.0)

    def test_simple_error(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.8])

        loss = mse_loss(y_true, y_pred)

        expected = (0.01 + 0.04 + 0.04) / 3
        np.testing.assert_allclose(loss, expected)

    def test_large_error(self) -> None:
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])

        loss = mse_loss(y_true, y_pred)

        expected = (9 + 16) / 2
        np.testing.assert_allclose(loss, expected)


class TestMSEGradient:
    def test_zero_gradient(self) -> None:
        X = np.array([[1, 2], [3, 4]])
        y_true = np.array([5, 11])
        y_pred = np.array([5, 11])

        grad_w, grad_b = mse_gradient(X, y_true, y_pred)

        np.testing.assert_allclose(grad_w, [0, 0], atol=1e-10)
        np.testing.assert_allclose(grad_b, 0.0, atol=1e-10)

    def test_simple_gradient(self) -> None:
        X = np.array([[1, 2], [3, 4]])
        y_true = np.array([5, 11])
        y_pred = np.array([5.5, 11.5])

        grad_w, grad_b = mse_gradient(X, y_true, y_pred)

        expected_grad_w = np.array([2.0, 3.0])
        expected_grad_b = 0.5

        np.testing.assert_allclose(grad_w, expected_grad_w)
        np.testing.assert_allclose(grad_b, expected_grad_b)

    def test_negative_error(self) -> None:
        X = np.array([[1], [2]])
        y_true = np.array([10, 20])
        y_pred = np.array([9, 19])

        grad_w, grad_b = mse_gradient(X, y_true, y_pred)

        assert grad_w[0] < 0
        assert grad_b < 0


class TestR2Score:
    def test_perfect_fit(self) -> None:
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        r2 = r2_score(y_true, y_pred)

        np.testing.assert_allclose(r2, 1.0)

    def test_mean_baseline(self) -> None:
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3])

        r2 = r2_score(y_true, y_pred)

        np.testing.assert_allclose(r2, 0.0, atol=1e-10)

    def test_negative_r2(self) -> None:
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([5, 4, 3, 2, 1])

        r2 = r2_score(y_true, y_pred)

        assert r2 < 0

    def test_constant_y(self) -> None:
        y_true = np.array([2.0, 2.0, 2.0])
        y_pred = np.array([2.0, 2.0, 2.0])

        r2 = r2_score(y_true, y_pred)

        np.testing.assert_allclose(r2, 0.0)


class TestNormalEquation:
    def test_simple_line(self) -> None:
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])

        weights, bias = normal_equation(X, y)

        np.testing.assert_allclose(weights, [2.0], atol=1e-6)
        np.testing.assert_allclose(bias, 0.0, atol=1e-6)

    def test_with_bias(self) -> None:
        X = np.array([[1], [2], [3]])
        y = np.array([3, 5, 7])

        weights, bias = normal_equation(X, y)

        np.testing.assert_allclose(weights, [2.0], atol=1e-6)
        np.testing.assert_allclose(bias, 1.0, atol=1e-6)

    def test_multivariate(self) -> None:
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.array([6, 8, 10, 12])

        weights, bias = normal_equation(X, y)

        y_pred = X @ weights + bias
        np.testing.assert_allclose(y_pred, y, atol=1e-6)


class TestStandardize:
    def test_simple_standardization(self) -> None:
        X = np.array([[1, 100], [2, 200], [3, 300]])

        X_scaled, mean, std = standardize(X)

        np.testing.assert_allclose(np.mean(X_scaled, axis=0), [0, 0], atol=1e-10)
        np.testing.assert_allclose(np.std(X_scaled, axis=0), [1, 1], atol=1e-10)

    def test_mean_and_std_returned(self) -> None:
        X = np.array([[1, 10], [2, 20], [3, 30]])

        X_scaled, mean, std = standardize(X)

        np.testing.assert_allclose(mean, [2, 20])
        assert std[0] > 0 and std[1] > 0

    def test_constant_feature(self) -> None:
        X = np.array([[1, 5], [2, 5], [3, 5]])

        X_scaled, mean, std = standardize(X)

        assert not np.any(np.isnan(X_scaled))
        assert not np.any(np.isinf(X_scaled))

    def test_inverse_transform(self) -> None:
        X = np.array([[1, 2], [3, 4], [5, 6]])

        X_scaled, mean, std = standardize(X)
        X_recovered = X_scaled * std + mean

        np.testing.assert_allclose(X_recovered, X, atol=1e-10)


class TestTrainTestSplit:
    def test_basic_split(self) -> None:
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_different_test_size(self) -> None:
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        assert len(X_train) == 70
        assert len(X_test) == 30

    def test_correspondence(self) -> None:
        X = np.arange(10).reshape(10, 1)
        y = np.arange(10) * 2

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        for i in range(len(X_train)):
            assert y_train[i] == X_train[i, 0] * 2


class TestPolynomialFeatures:
    def test_degree_2_single_feature(self) -> None:
        X = np.array([[2], [3]])

        X_poly = polynomial_features(X, degree=2)

        expected = np.array([[2, 4], [3, 9]])
        np.testing.assert_allclose(X_poly, expected)

    def test_degree_3_single_feature(self) -> None:
        X = np.array([[2]])

        X_poly = polynomial_features(X, degree=3)

        expected = np.array([[2, 4, 8]])
        np.testing.assert_allclose(X_poly, expected)

    def test_degree_2_two_features(self) -> None:
        X = np.array([[2, 3]])

        X_poly = polynomial_features(X, degree=2)

        assert X_poly.shape[1] == 5

    def test_degree_1(self) -> None:
        X = np.array([[1, 2], [3, 4]])

        X_poly = polynomial_features(X, degree=1)

        np.testing.assert_allclose(X_poly, X)


class TestComposition:
    def test_full_pipeline(self) -> None:
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        true_weights = np.array([1.5, -2.0])
        true_bias = 0.5
        y = X @ true_weights + true_bias + np.random.randn(n) * 0.1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        X_train_scaled, mean, std = standardize(X_train)
        X_test_scaled = (X_test - mean) / std

        weights, bias = normal_equation(X_train_scaled, y_train)

        y_test_pred = predict(X_test_scaled, weights, bias)

        test_loss = mse_loss(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        assert test_loss < 0.1
        assert test_r2 > 0.9

    def test_gradient_descent_loop(self) -> None:
        np.random.seed(42)
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10], dtype=float)

        X_scaled, mean, std = standardize(X)

        weights = np.zeros(1)
        bias = 0.0
        learning_rate = 0.1

        for _ in range(1000):
            y_pred = predict(X_scaled, weights, bias)
            grad_w, grad_b = mse_gradient(X_scaled, y, y_pred)

            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b

        y_pred_final = predict(X_scaled, weights, bias)
        final_r2 = r2_score(y, y_pred_final)

        assert final_r2 > 0.99
