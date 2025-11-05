from __future__ import annotations

import numpy as np


def binary_cross_entropy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute binary cross-entropy loss.

    Args:
        y_true: True labels {0, 1}, shape (n_samples,)
        y_pred_proba: Predicted probabilities, shape (n_samples,)

    Returns:
        loss: Binary cross-entropy loss (scalar)
    """
    raise NotImplementedError


def categorical_cross_entropy(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> float:
    """
    Compute categorical cross-entropy loss.

    Args:
        y_true: True labels (one-hot), shape (n_samples, n_classes)
        y_pred_proba: Predicted probabilities, shape (n_samples, n_classes)

    Returns:
        loss: Categorical cross-entropy loss (scalar)
    """
    raise NotImplementedError


def binary_cross_entropy_gradient(
    X: np.ndarray, y_true: np.ndarray, y_pred_proba: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Compute gradients of binary cross-entropy loss.

    Args:
        X: Features, shape (n_samples, n_features)
        y_true: True labels {0, 1}, shape (n_samples,)
        y_pred_proba: Predicted probabilities, shape (n_samples,)

    Returns:
        grad_w: Gradient w.r.t. weights, shape (n_features,)
        grad_b: Gradient w.r.t. bias (scalar)
    """
    raise NotImplementedError


def categorical_cross_entropy_gradient(
    X: np.ndarray, y_true: np.ndarray, y_pred_proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute gradients of categorical cross-entropy loss.

    Args:
        X: Features, shape (n_samples, n_features)
        y_true: True labels (one-hot), shape (n_samples, n_classes)
        y_pred_proba: Predicted probabilities, shape (n_samples, n_classes)

    Returns:
        grad_w: Gradient w.r.t. weights, shape (n_features, n_classes)
        grad_b: Gradient w.r.t. bias, shape (n_classes,)
    """
    raise NotImplementedError
