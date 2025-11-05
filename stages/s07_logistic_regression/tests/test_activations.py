from __future__ import annotations

import numpy as np
import pytest

from stages.s07_logistic_regression.starter.activations import (
    predict_binary,
    predict_multiclass,
    predict_proba_binary,
    predict_proba_multiclass,
    sigmoid,
    softmax,
)


class TestSigmoid:
    def test_zero_input(self):
        z = np.array([0.0])
        result = sigmoid(z)
        np.testing.assert_allclose(result, [0.5])

    def test_positive_values(self):
        z = np.array([1.0, 2.0, 10.0])
        result = sigmoid(z)
        expected = 1 / (1 + np.exp(-z))
        np.testing.assert_allclose(result, expected)

    def test_negative_values(self):
        z = np.array([-1.0, -2.0, -10.0])
        result = sigmoid(z)
        expected = 1 / (1 + np.exp(-z))
        np.testing.assert_allclose(result, expected)

    def test_large_values_no_overflow(self):
        z = np.array([100, 500, 1000])
        result = sigmoid(z)
        assert np.all(result > 0.999)
        assert np.all(~np.isnan(result))
        assert np.all(~np.isinf(result))

    def test_large_negative_values_no_underflow(self):
        z = np.array([-100, -500, -1000])
        result = sigmoid(z)
        assert np.all(result < 0.001)
        assert np.all(~np.isnan(result))
        assert np.all(~np.isinf(result))

    def test_symmetry(self):
        z = np.array([1.0, 2.0, 3.0])
        assert np.allclose(sigmoid(z) + sigmoid(-z), 1.0)


class TestSoftmax:
    def test_two_classes(self):
        z = np.array([[1.0, 2.0]])
        result = softmax(z)
        assert np.allclose(result.sum(axis=1), [1.0])
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_three_classes(self):
        z = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        result = softmax(z)
        np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0])

    def test_uniform_inputs(self):
        z = np.array([[1.0, 1.0, 1.0]])
        result = softmax(z)
        expected = np.array([[1/3, 1/3, 1/3]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_numerical_stability(self):
        z = np.array([[1000, 1001, 1002]])
        result = softmax(z)
        assert np.all(~np.isnan(result))
        assert np.all(~np.isinf(result))
        assert np.allclose(result.sum(axis=1), [1.0])

    def test_preserves_argmax(self):
        z = np.array([[1.0, 5.0, 2.0], [3.0, 1.0, 2.0]])
        result = softmax(z)
        assert np.argmax(result[0]) == 1
        assert np.argmax(result[1]) == 0


class TestPredictProbaBinary:
    def test_simple_prediction(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        weights = np.array([0.5, 0.5])
        bias = 0.0

        probs = predict_proba_binary(X, weights, bias)

        z = X @ weights + bias
        expected = 1 / (1 + np.exp(-z))
        np.testing.assert_allclose(probs, expected)

    def test_with_bias(self):
        X = np.array([[0.0, 0.0]])
        weights = np.array([1.0, 1.0])
        bias = 2.0

        probs = predict_proba_binary(X, weights, bias)

        expected = sigmoid(np.array([2.0]))
        np.testing.assert_allclose(probs, expected)

    def test_shape(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        weights = np.array([0.5, 0.5])
        bias = 0.0

        probs = predict_proba_binary(X, weights, bias)

        assert probs.shape == (3,)


class TestPredictProbaMulticlass:
    def test_two_classes(self):
        X = np.array([[1.0, 2.0]])
        weights = np.array([[0.5, 0.3], [0.2, 0.4]])
        bias = np.array([0.0, 0.1])

        probs = predict_proba_multiclass(X, weights, bias)

        assert probs.shape == (1, 2)
        assert np.allclose(probs.sum(axis=1), [1.0])

    def test_three_classes(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        weights = np.array([[0.5, 0.3, 0.2], [0.2, 0.4, 0.1]])
        bias = np.array([0.0, 0.0, 0.0])

        probs = predict_proba_multiclass(X, weights, bias)

        assert probs.shape == (2, 3)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])


class TestPredictBinary:
    def test_default_threshold(self):
        X = np.array([[1.0, 2.0], [0.0, 0.0], [-1.0, -2.0]])
        weights = np.array([1.0, 1.0])
        bias = 0.0

        preds = predict_binary(X, weights, bias)

        assert preds[0] == 1
        assert preds[1] == 0
        assert preds[2] == 0

    def test_custom_threshold(self):
        X = np.array([[1.0, 1.0]])
        weights = np.array([0.5, 0.5])
        bias = 0.0

        preds_low = predict_binary(X, weights, bias, threshold=0.1)
        preds_high = predict_binary(X, weights, bias, threshold=0.9)

        assert preds_low[0] == 1
        assert preds_high[0] in [0, 1]

    def test_output_type(self):
        X = np.array([[1.0, 2.0]])
        weights = np.array([0.5, 0.5])
        bias = 0.0

        preds = predict_binary(X, weights, bias)

        assert preds.dtype in [np.int32, np.int64, int]


class TestPredictMulticlass:
    def test_simple_prediction(self):
        X = np.array([[1.0, 2.0]])
        weights = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        bias = np.array([0.0, 0.0, 0.0])

        preds = predict_multiclass(X, weights, bias)

        assert preds.shape == (1,)
        assert preds[0] in [0, 1, 2]

    def test_three_samples(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        weights = np.array([[2.0, 0.0], [0.0, 2.0]])
        bias = np.array([0.0, 0.0])

        preds = predict_multiclass(X, weights, bias)

        assert preds.shape == (3,)
        assert preds[0] == 0
        assert preds[1] == 1
