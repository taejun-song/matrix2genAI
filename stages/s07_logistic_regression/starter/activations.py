from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Apply sigmoid activation function.

    Args:
        z: Linear outputs, any shape

    Returns:
        probabilities: Sigmoid applied element-wise, same shape as z
    """
    raise NotImplementedError


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Apply softmax activation function.

    Args:
        z: Linear outputs, shape (n_samples, n_classes)

    Returns:
        probabilities: Softmax outputs, shape (n_samples, n_classes)
    """
    raise NotImplementedError


def predict_proba_binary(
    X: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    """
    Compute binary class probabilities.

    Args:
        X: Features, shape (n_samples, n_features)
        weights: Weight vector, shape (n_features,)
        bias: Bias term (scalar)

    Returns:
        probabilities: P(y=1|X), shape (n_samples,)
    """
    raise NotImplementedError


def predict_proba_multiclass(
    X: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """
    Compute multi-class probabilities.

    Args:
        X: Features, shape (n_samples, n_features)
        weights: Weight matrix, shape (n_features, n_classes)
        bias: Bias vector, shape (n_classes,)

    Returns:
        probabilities: Class probabilities, shape (n_samples, n_classes)
    """
    raise NotImplementedError


def predict_binary(
    X: np.ndarray, weights: np.ndarray, bias: float, threshold: float = 0.5
) -> np.ndarray:
    """
    Make binary predictions.

    Args:
        X: Features, shape (n_samples, n_features)
        weights: Weight vector, shape (n_features,)
        bias: Bias term (scalar)
        threshold: Decision threshold

    Returns:
        predictions: Binary labels {0, 1}, shape (n_samples,)
    """
    raise NotImplementedError


def predict_multiclass(
    X: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """
    Make multi-class predictions.

    Args:
        X: Features, shape (n_samples, n_features)
        weights: Weight matrix, shape (n_features, n_classes)
        bias: Bias vector, shape (n_classes,)

    Returns:
        predictions: Class indices, shape (n_samples,)
    """
    raise NotImplementedError
