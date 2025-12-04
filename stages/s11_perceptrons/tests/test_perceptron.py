from __future__ import annotations

import numpy as np

from stages.s11_perceptrons.starter.perceptron import (
    perceptron_forward,
    perceptron_predict,
    perceptron_train,
    perceptron_update,
)


class TestPerceptronForward:
    def test_positive_output(self) -> None:
        x = np.array([1.0, 1.0])
        w = np.array([1.0, 1.0])
        b = 0.0
        result = perceptron_forward(x, w, b)
        assert result == 1

    def test_negative_output(self) -> None:
        x = np.array([1.0, 1.0])
        w = np.array([-1.0, -1.0])
        b = 0.0
        result = perceptron_forward(x, w, b)
        assert result == 0

    def test_bias_effect(self) -> None:
        x = np.array([0.0, 0.0])
        w = np.array([1.0, 1.0])
        assert perceptron_forward(x, w, bias=1.0) == 1
        assert perceptron_forward(x, w, bias=-1.0) == 0

    def test_zero_threshold(self) -> None:
        x = np.array([1.0])
        w = np.array([0.0])
        b = 0.0
        result = perceptron_forward(x, w, b)
        assert result == 0


class TestPerceptronUpdate:
    def test_correct_prediction_no_update(self) -> None:
        x = np.array([1.0, 1.0])
        w = np.array([1.0, 1.0])
        b = 0.0
        new_w, new_b = perceptron_update(x, y=1, weights=w, bias=b)
        np.testing.assert_array_equal(new_w, w)
        assert new_b == b

    def test_wrong_prediction_updates(self) -> None:
        x = np.array([1.0, 1.0])
        w = np.array([0.0, 0.0])
        b = 0.0
        new_w, new_b = perceptron_update(x, y=1, weights=w, bias=b, lr=1.0)
        np.testing.assert_array_equal(new_w, [1.0, 1.0])
        assert new_b == 1.0

    def test_learning_rate(self) -> None:
        x = np.array([1.0, 1.0])
        w = np.array([0.0, 0.0])
        b = 0.0
        new_w, new_b = perceptron_update(x, y=1, weights=w, bias=b, lr=0.5)
        np.testing.assert_array_equal(new_w, [0.5, 0.5])
        assert new_b == 0.5

    def test_negative_update(self) -> None:
        x = np.array([1.0, 1.0])
        w = np.array([1.0, 1.0])
        b = 1.0
        new_w, new_b = perceptron_update(x, y=0, weights=w, bias=b, lr=1.0)
        np.testing.assert_array_equal(new_w, [0.0, 0.0])
        assert new_b == 0.0


class TestPerceptronTrain:
    def test_and_gate(self) -> None:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
        w, b, history = perceptron_train(X, y, n_epochs=20)
        predictions = perceptron_predict(X, w, b)
        np.testing.assert_array_equal(predictions, y)

    def test_or_gate(self) -> None:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        w, b, history = perceptron_train(X, y, n_epochs=20)
        predictions = perceptron_predict(X, w, b)
        np.testing.assert_array_equal(predictions, y)

    def test_convergence_history(self) -> None:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
        _, _, history = perceptron_train(X, y, n_epochs=100)
        assert history[-1] == 0

    def test_xor_does_not_converge(self) -> None:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        w, b, history = perceptron_train(X, y, n_epochs=100)
        predictions = perceptron_predict(X, w, b)
        accuracy = np.mean(predictions == y)
        assert accuracy < 1.0

    def test_returns_weights_bias_history(self) -> None:
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        result = perceptron_train(X, y, n_epochs=10)
        assert len(result) == 3
        w, b, h = result
        assert w.shape == (2,)
        assert isinstance(b, float)
        assert isinstance(h, list)


class TestPerceptronPredict:
    def test_batch_prediction(self) -> None:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        w = np.array([1.0, 1.0])
        b = -1.5
        predictions = perceptron_predict(X, w, b)
        np.testing.assert_array_equal(predictions, [0, 0, 0, 1])

    def test_output_shape(self) -> None:
        X = np.array([[1, 2], [3, 4], [5, 6]])
        w = np.array([0.5, 0.5])
        b = 0.0
        predictions = perceptron_predict(X, w, b)
        assert predictions.shape == (3,)

    def test_binary_output(self) -> None:
        np.random.seed(42)
        X = np.random.randn(100, 5)
        w = np.random.randn(5)
        b = 0.0
        predictions = perceptron_predict(X, w, b)
        assert set(predictions).issubset({0, 1})


class TestLinearSeparability:
    def test_linearly_separable_converges(self) -> None:
        np.random.seed(42)
        X_class0 = np.random.randn(20, 2) + np.array([-2, -2])
        X_class1 = np.random.randn(20, 2) + np.array([2, 2])
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * 20 + [1] * 20)
        w, b, history = perceptron_train(X, y, n_epochs=100)
        predictions = perceptron_predict(X, w, b)
        accuracy = np.mean(predictions == y)
        assert accuracy == 1.0

    def test_nand_gate(self) -> None:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([1, 1, 1, 0])
        w, b, _ = perceptron_train(X, y, n_epochs=20)
        predictions = perceptron_predict(X, w, b)
        np.testing.assert_array_equal(predictions, y)
