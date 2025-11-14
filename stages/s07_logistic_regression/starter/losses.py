from __future__ import annotations

import numpy as np


# ============================================================================
# Building Block 5: Cross-Entropy Loss Components
# ============================================================================


def _clip_probabilities(proba: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    """
    Clip probabilities to avoid log(0).

    Args:
        proba: Probabilities, any shape
        epsilon: Small value for clipping

    Returns:
        Clipped probabilities in [epsilon, 1-epsilon]

    Example:
        >>> _clip_probabilities(np.array([0.0, 0.5, 1.0]))
        array([1e-15, 0.5, 0.999...])
    """
    # TODO: Clip values to [epsilon, 1-epsilon]
    # Use np.clip(proba, epsilon, 1 - epsilon)
    raise NotImplementedError


def _binary_cross_entropy_per_sample(y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
    """
    Compute binary cross-entropy per sample (before averaging).

    Formula: -[y*log(p) + (1-y)*log(1-p)]

    Args:
        y_true: True labels {0, 1}, shape (n_samples,)
        y_pred_proba: Predicted probabilities, shape (n_samples,)

    Returns:
        Per-sample losses, shape (n_samples,)

    Example:
        >>> y = np.array([1, 0, 1])
        >>> p = np.array([0.9, 0.1, 0.8])
        >>> _binary_cross_entropy_per_sample(y, p)
        array([0.105, 0.105, 0.223])
    """
    # TODO: Compute per-sample loss
    # 1. p_safe = _clip_probabilities(y_pred_proba)
    # 2. return -(y_true * np.log(p_safe) + (1 - y_true) * np.log(1 - p_safe))
    raise NotImplementedError


def binary_cross_entropy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute binary cross-entropy loss: L = -(1/n)Σ[y*log(p) + (1-y)*log(1-p)]

    Args:
        y_true: True labels {0, 1}, shape (n_samples,)
        y_pred_proba: Predicted probabilities, shape (n_samples,)

    Returns:
        loss: Average binary cross-entropy (scalar)

    Example:
        >>> y = np.array([1, 0, 1, 0])
        >>> p = np.array([0.9, 0.1, 0.8, 0.2])
        >>> binary_cross_entropy(y, p)
        0.174
    """
    # TODO: Average per-sample losses
    # losses = _binary_cross_entropy_per_sample(y_true, y_pred_proba)
    # return float(np.mean(losses))
    raise NotImplementedError


def categorical_cross_entropy(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> float:
    """
    Compute categorical cross-entropy: L = -(1/n)Σᵢ Σⱼ yᵢⱼ * log(pᵢⱼ)

    Args:
        y_true: True labels (one-hot), shape (n_samples, n_classes)
        y_pred_proba: Predicted probabilities, shape (n_samples, n_classes)

    Returns:
        loss: Average categorical cross-entropy (scalar)

    Example:
        >>> y = np.array([[1, 0, 0], [0, 1, 0]])
        >>> p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        >>> categorical_cross_entropy(y, p)
        0.188
    """
    # TODO: Compute categorical cross-entropy
    # 1. p_safe = _clip_probabilities(y_pred_proba)
    # 2. per_sample = -np.sum(y_true * np.log(p_safe), axis=1)
    # 3. return float(np.mean(per_sample))
    raise NotImplementedError


# ============================================================================
# Building Block 6: Gradient Components
# ============================================================================


def _compute_error(y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
    """
    Compute prediction error: error = y_pred - y_true

    This is the key quantity for gradients in logistic regression!

    Args:
        y_true: True labels, any shape
        y_pred_proba: Predicted probabilities, same shape

    Returns:
        error: Prediction errors, same shape

    Example:
        >>> _compute_error(np.array([1, 0, 1]), np.array([0.9, 0.2, 0.7]))
        array([-0.1, 0.2, -0.3])
    """
    # TODO: Compute prediction error
    # return y_pred_proba - y_true
    raise NotImplementedError


def binary_cross_entropy_gradient(
    X: np.ndarray, y_true: np.ndarray, y_pred_proba: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Compute gradients: ∂L/∂w = (1/n)X^T(p-y), ∂L/∂b = (1/n)Σ(p-y)

    Args:
        X: Features, shape (n_samples, n_features)
        y_true: True labels {0, 1}, shape (n_samples,)
        y_pred_proba: Predicted probabilities, shape (n_samples,)

    Returns:
        grad_w: Gradient w.r.t. weights, shape (n_features,)
        grad_b: Gradient w.r.t. bias (scalar)

    Example:
        >>> X = np.array([[1, 2], [3, 4]])
        >>> y = np.array([1, 0])
        >>> p = np.array([0.8, 0.3])
        >>> grad_w, grad_b = binary_cross_entropy_gradient(X, y, p)
        >>> grad_w.shape
        (2,)
    """
    # TODO: Compute gradients
    # 1. n = len(y_true)
    # 2. error = _compute_error(y_true, y_pred_proba)
    # 3. grad_w = (1/n) * X.T @ error
    # 4. grad_b = (1/n) * np.sum(error)
    # 5. return grad_w, float(grad_b)
    raise NotImplementedError


def categorical_cross_entropy_gradient(
    X: np.ndarray, y_true: np.ndarray, y_pred_proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute gradients: ∂L/∂W = (1/n)X^T(P-Y), ∂L/∂b = (1/n)Σ(P-Y)

    Args:
        X: Features, shape (n_samples, n_features)
        y_true: True labels (one-hot), shape (n_samples, n_classes)
        y_pred_proba: Predicted probabilities, shape (n_samples, n_classes)

    Returns:
        grad_w: Gradient w.r.t. weights, shape (n_features, n_classes)
        grad_b: Gradient w.r.t. bias, shape (n_classes,)

    Example:
        >>> X = np.array([[1, 2], [3, 4]])
        >>> Y = np.array([[1, 0, 0], [0, 1, 0]])
        >>> P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        >>> grad_w, grad_b = categorical_cross_entropy_gradient(X, Y, P)
        >>> grad_w.shape
        (2, 3)
    """
    # TODO: Compute gradients (same formula as binary!)
    # 1. n = len(y_true)
    # 2. error = _compute_error(y_true, y_pred_proba)
    # 3. grad_w = (1/n) * X.T @ error
    # 4. grad_b = (1/n) * np.sum(error, axis=0)
    # 5. return grad_w, grad_b
    raise NotImplementedError
