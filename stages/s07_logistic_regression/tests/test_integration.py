from __future__ import annotations

import numpy as np

from stages.s07_logistic_regression.starter.activations import (
    predict_binary,
    predict_multiclass,
    predict_proba_binary,
    predict_proba_multiclass,
)
from stages.s07_logistic_regression.starter.losses import (
    binary_cross_entropy,
    binary_cross_entropy_gradient,
    categorical_cross_entropy,
    categorical_cross_entropy_gradient,
)
from stages.s07_logistic_regression.starter.metrics import (
    accuracy,
    classification_report,
)


class TestBinaryClassificationPipeline:
    def test_simple_binary_training(self):
        np.random.seed(42)

        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        weights = np.zeros(2)
        bias = 0.0
        learning_rate = 0.1
        num_epochs = 100

        initial_loss = None
        final_loss = None

        for epoch in range(num_epochs):
            y_pred_proba = predict_proba_binary(X, weights, bias)
            loss = binary_cross_entropy(y, y_pred_proba)

            if epoch == 0:
                initial_loss = loss

            grad_w, grad_b = binary_cross_entropy_gradient(X, y, y_pred_proba)

            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b

            if epoch == num_epochs - 1:
                final_loss = loss

        assert final_loss < initial_loss
        assert final_loss < 0.5

        y_pred = predict_binary(X, weights, bias)
        acc = accuracy(y, y_pred)

        assert acc > 0.7

    def test_binary_gradient_descent_decreases_loss(self):
        np.random.seed(42)

        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)

        weights = np.random.randn(2) * 0.01
        bias = 0.0
        learning_rate = 0.1

        losses = []

        for _epoch in range(50):
            y_pred_proba = predict_proba_binary(X, weights, bias)
            loss = binary_cross_entropy(y, y_pred_proba)
            losses.append(loss)

            grad_w, grad_b = binary_cross_entropy_gradient(X, y, y_pred_proba)

            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b

        assert losses[-1] < losses[0]
        assert all(not np.isnan(loss) for loss in losses)

    def test_binary_classification_report(self):
        np.random.seed(42)

        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        weights = np.zeros(2)
        bias = 0.0
        learning_rate = 0.1

        for _ in range(100):
            y_pred_proba = predict_proba_binary(X, weights, bias)
            grad_w, grad_b = binary_cross_entropy_gradient(X, y, y_pred_proba)
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b

        y_pred_proba = predict_proba_binary(X, weights, bias)
        y_pred = predict_binary(X, weights, bias)

        report = classification_report(y, y_pred, y_pred_proba)

        assert report["accuracy"] > 0.7
        assert 0 <= report["roc_auc"] <= 1


class TestMulticlassClassificationPipeline:
    def test_simple_multiclass_training(self):
        np.random.seed(42)

        X = np.random.randn(150, 2)
        y_labels = np.zeros(150, dtype=int)
        y_labels[50:100] = 1
        y_labels[100:] = 2

        y_onehot = np.eye(3)[y_labels]

        n_features = 2
        n_classes = 3
        weights = np.zeros((n_features, n_classes))
        bias = np.zeros(n_classes)
        learning_rate = 0.1
        num_epochs = 100

        initial_loss = None
        final_loss = None

        for epoch in range(num_epochs):
            y_pred_proba = predict_proba_multiclass(X, weights, bias)
            loss = categorical_cross_entropy(y_onehot, y_pred_proba)

            if epoch == 0:
                initial_loss = loss

            grad_w, grad_b = categorical_cross_entropy_gradient(X, y_onehot, y_pred_proba)

            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b

            if epoch == num_epochs - 1:
                final_loss = loss

        assert final_loss < initial_loss

        y_pred = predict_multiclass(X, weights, bias)
        acc = accuracy(y_labels, y_pred)

        assert acc > 0.3

    def test_multiclass_gradient_descent(self):
        np.random.seed(42)

        X = np.random.randn(90, 2)
        y_labels = np.zeros(90, dtype=int)
        y_labels[30:60] = 1
        y_labels[60:] = 2

        y_onehot = np.eye(3)[y_labels]

        weights = np.random.randn(2, 3) * 0.01
        bias = np.zeros(3)
        learning_rate = 0.1

        losses = []

        for _epoch in range(50):
            y_pred_proba = predict_proba_multiclass(X, weights, bias)
            loss = categorical_cross_entropy(y_onehot, y_pred_proba)
            losses.append(loss)

            grad_w, grad_b = categorical_cross_entropy_gradient(X, y_onehot, y_pred_proba)

            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b

        assert losses[-1] < losses[0]
        assert all(not np.isnan(loss) for loss in losses)


class TestNumericalStability:
    def test_sigmoid_large_values(self):
        from stages.s07_logistic_regression.starter.activations import sigmoid

        z = np.array([1000, -1000, 500, -500])
        result = sigmoid(z)

        assert np.all(~np.isnan(result))
        assert np.all(~np.isinf(result))
        assert np.all((result >= 0) & (result <= 1))

    def test_softmax_large_values(self):
        from stages.s07_logistic_regression.starter.activations import softmax

        z = np.array([[1000, 1001, 1002], [-1000, -1001, -1002]])
        result = softmax(z)

        assert np.all(~np.isnan(result))
        assert np.all(~np.isinf(result))
        assert np.allclose(result.sum(axis=1), [1.0, 1.0])

    def test_binary_cross_entropy_extreme_probabilities(self):
        y_true = np.array([1, 0])
        y_pred_proba = np.array([0.99999, 0.00001])

        loss = binary_cross_entropy(y_true, y_pred_proba)

        assert np.isfinite(loss)
        assert loss >= 0

    def test_categorical_cross_entropy_extreme_probabilities(self):
        y_true = np.array([[1, 0, 0]])
        y_pred_proba = np.array([[0.99999, 0.000005, 0.000005]])

        loss = categorical_cross_entropy(y_true, y_pred_proba)

        assert np.isfinite(loss)
        assert loss >= 0
