from __future__ import annotations

import numpy as np


def perceptron_forward(x: np.ndarray, weights: np.ndarray, bias: float) -> int:
    """
    Single perceptron forward pass.

    Args:
        x: Input vector, shape (n_features,)
        weights: Weight vector, shape (n_features,)
        bias: Bias term (scalar)

    Returns:
        prediction: 0 or 1

    Example:
        >>> x = np.array([1.0, 2.0])
        >>> w = np.array([0.5, -0.5])
        >>> perceptron_forward(x, w, bias=0.0)
        0
    """
    # TODO:
    # z = np.dot(x, weights) + bias
    # return 1 if z > 0 else 0
    raise NotImplementedError


def perceptron_update(
    x: np.ndarray,
    y: int,
    weights: np.ndarray,
    bias: float,
    lr: float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    Update perceptron weights if prediction is wrong.

    Args:
        x: Input vector, shape (n_features,)
        y: True label (0 or 1)
        weights: Current weights, shape (n_features,)
        bias: Current bias
        lr: Learning rate

    Returns:
        new_weights: Updated weights
        new_bias: Updated bias

    Example:
        >>> x = np.array([1.0, 1.0])
        >>> w = np.array([0.0, 0.0])
        >>> new_w, new_b = perceptron_update(x, y=1, weights=w, bias=0.0, lr=1.0)
        >>> new_w  # Should be updated since prediction was 0, not 1
        array([1., 1.])
    """
    # TODO:
    # y_pred = perceptron_forward(x, weights, bias)
    # if y_pred != y:
    #     weights = weights + lr * (y - y_pred) * x
    #     bias = bias + lr * (y - y_pred)
    # return weights, bias
    raise NotImplementedError


def perceptron_train(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 1.0,
    n_epochs: int = 100,
) -> tuple[np.ndarray, float, list[int]]:
    """
    Train perceptron on dataset.

    Args:
        X: Features, shape (n_samples, n_features)
        y: Labels (0 or 1), shape (n_samples,)
        lr: Learning rate
        n_epochs: Maximum number of epochs

    Returns:
        weights: Trained weights
        bias: Trained bias
        history: List of error counts per epoch

    Example:
        >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        >>> y = np.array([0, 0, 0, 1])  # AND gate
        >>> w, b, hist = perceptron_train(X, y, n_epochs=10)
        >>> hist[-1]  # Should converge to 0 errors
        0
    """
    # TODO:
    # n_features = X.shape[1]
    # weights = np.zeros(n_features)
    # bias = 0.0
    # history = []
    #
    # for epoch in range(n_epochs):
    #     errors = 0
    #     for i in range(len(X)):
    #         y_pred = perceptron_forward(X[i], weights, bias)
    #         if y_pred != y[i]:
    #             errors += 1
    #         weights, bias = perceptron_update(X[i], y[i], weights, bias, lr)
    #     history.append(errors)
    #     if errors == 0:
    #         break
    #
    # return weights, bias, history
    raise NotImplementedError


def perceptron_predict(
    X: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    """
    Batch perceptron predictions.

    Args:
        X: Features, shape (n_samples, n_features)
        weights: Trained weights, shape (n_features,)
        bias: Trained bias

    Returns:
        predictions: Array of 0/1, shape (n_samples,)

    Example:
        >>> X = np.array([[0, 0], [1, 1]])
        >>> w = np.array([1.0, 1.0])
        >>> perceptron_predict(X, w, bias=-1.5)
        array([0, 1])
    """
    # TODO:
    # z = X @ weights + bias
    # return (z > 0).astype(int)
    raise NotImplementedError
