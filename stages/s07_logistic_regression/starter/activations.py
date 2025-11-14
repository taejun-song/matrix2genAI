from __future__ import annotations

import numpy as np


# ============================================================================
# Building Block 1: Numerically Stable Sigmoid Components
# ============================================================================


def _sigmoid_positive(z: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid for positive values: σ(z) = 1 / (1 + e^(-z))

    This is numerically stable for z >= 0.

    Args:
        z: Input values (should be >= 0), any shape

    Returns:
        Sigmoid values, same shape as z

    Example:
        >>> _sigmoid_positive(np.array([0.0, 1.0, 100.0]))
        array([0.5, 0.731, 1.0])
    """
    # TODO: Implement for positive z
    # Formula: 1 / (1 + np.exp(-z))
    raise NotImplementedError


def _sigmoid_negative(z: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid for negative values: σ(z) = e^z / (1 + e^z)

    This is numerically stable for z < 0.

    Args:
        z: Input values (should be < 0), any shape

    Returns:
        Sigmoid values, same shape as z

    Example:
        >>> _sigmoid_negative(np.array([-1.0, -100.0]))
        array([0.269, 0.0])
    """
    # TODO: Implement for negative z
    # Formula: exp_z = np.exp(z); return exp_z / (1 + exp_z)
    raise NotImplementedError


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Apply numerically stable sigmoid: σ(z) = 1 / (1 + e^(-z))

    Args:
        z: Linear outputs, any shape

    Returns:
        probabilities: Sigmoid applied element-wise, same shape as z

    Example:
        >>> sigmoid(np.array([0, 1, -1, 100, -100]))
        array([0.5, 0.731, 0.269, 1.0, 0.0])
    """
    # TODO: Combine _sigmoid_positive and _sigmoid_negative
    # Use np.where(z >= 0, _sigmoid_positive(z), _sigmoid_negative(z))
    raise NotImplementedError


# ============================================================================
# Building Block 2: Numerically Stable Softmax Components
# ============================================================================


def _shift_by_max(z: np.ndarray) -> np.ndarray:
    """
    Shift z by subtracting max value per row for numerical stability.

    Args:
        z: Input values, shape (n_samples, n_classes)

    Returns:
        z_shifted: z with max subtracted per row, same shape

    Example:
        >>> _shift_by_max(np.array([[1, 2, 3], [10, 20, 30]]))
        array([[-2, -1, 0], [-20, -10, 0]])
    """
    # TODO: Subtract row-wise maximum
    # Use np.max(z, axis=1, keepdims=True) to get max per row
    # Then return z - max_values
    raise NotImplementedError


def _normalize_rows(exp_z: np.ndarray) -> np.ndarray:
    """
    Normalize rows to sum to 1.0.

    Args:
        exp_z: Exponential values, shape (n_samples, n_classes)

    Returns:
        probabilities: Normalized rows, same shape

    Example:
        >>> _normalize_rows(np.array([[1, 2, 3], [4, 4, 4]]))
        array([[0.167, 0.333, 0.5], [0.333, 0.333, 0.333]])
    """
    # TODO: Divide each row by its sum
    # Use np.sum(exp_z, axis=1, keepdims=True) to get row sums
    # Then return exp_z / row_sums
    raise NotImplementedError


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Apply numerically stable softmax: softmax(z)_j = exp(z_j) / Σ_k exp(z_k)

    Args:
        z: Linear outputs, shape (n_samples, n_classes)

    Returns:
        probabilities: Softmax outputs, shape (n_samples, n_classes)

    Example:
        >>> z = np.array([[1, 2, 3], [1, 1, 1]])
        >>> softmax(z)
        array([[0.09, 0.24, 0.67], [0.33, 0.33, 0.33]])
    """
    # TODO: Combine the sub-functions
    # 1. z_shifted = _shift_by_max(z)
    # 2. exp_z = np.exp(z_shifted)
    # 3. return _normalize_rows(exp_z)
    raise NotImplementedError


# ============================================================================
# Building Block 3: Linear Combination
# ============================================================================


def _linear_forward(X: np.ndarray, weights: np.ndarray, bias: float | np.ndarray) -> np.ndarray:
    """
    Compute linear combination: z = Xw + b

    Args:
        X: Features, shape (n_samples, n_features)
        weights: Weights, shape (n_features,) or (n_features, n_classes)
        bias: Bias, scalar or shape (n_classes,)

    Returns:
        z: Linear outputs, shape (n_samples,) or (n_samples, n_classes)

    Example:
        >>> X = np.array([[1, 2], [3, 4]])
        >>> w = np.array([0.5, -0.5])
        >>> b = 0.1
        >>> _linear_forward(X, w, b)
        array([-0.4, -0.4])
    """
    # TODO: Compute Xw + b
    # Use @ operator for matrix multiplication
    raise NotImplementedError


# ============================================================================
# Building Block 4: Predictions (composing previous functions)
# ============================================================================


def predict_proba_binary(
    X: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    """
    Compute binary class probabilities: P(y=1|X) = σ(Xw + b)

    Args:
        X: Features, shape (n_samples, n_features)
        weights: Weight vector, shape (n_features,)
        bias: Bias term (scalar)

    Returns:
        probabilities: P(y=1|X), shape (n_samples,)

    Example:
        >>> X = np.array([[1, 2], [3, 4]])
        >>> w = np.array([0.5, -0.5])
        >>> b = 0.1
        >>> predict_proba_binary(X, w, b)
        array([0.475, 0.475])
    """
    # TODO: Compose _linear_forward and sigmoid
    # z = _linear_forward(X, weights, bias)
    # return sigmoid(z)
    raise NotImplementedError


def predict_proba_multiclass(
    X: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """
    Compute multi-class probabilities: P(y=k|X) = softmax(Xw + b)

    Args:
        X: Features, shape (n_samples, n_features)
        weights: Weight matrix, shape (n_features, n_classes)
        bias: Bias vector, shape (n_classes,)

    Returns:
        probabilities: Class probabilities, shape (n_samples, n_classes)

    Example:
        >>> X = np.array([[1, 2], [3, 4]])
        >>> W = np.array([[0.5, -0.5, 0.1], [0.2, 0.3, -0.4]])
        >>> b = np.array([0.1, 0.2, 0.3])
        >>> probs = predict_proba_multiclass(X, W, b)
        >>> probs.shape
        (2, 3)
    """
    # TODO: Compose _linear_forward and softmax
    # z = _linear_forward(X, weights, bias)
    # return softmax(z)
    raise NotImplementedError


def predict_binary(
    X: np.ndarray, weights: np.ndarray, bias: float, threshold: float = 0.5
) -> np.ndarray:
    """
    Make binary predictions: ŷ = 1 if P(y=1|X) >= threshold else 0

    Args:
        X: Features, shape (n_samples, n_features)
        weights: Weight vector, shape (n_features,)
        bias: Bias term (scalar)
        threshold: Decision threshold

    Returns:
        predictions: Binary labels {0, 1}, shape (n_samples,)

    Example:
        >>> X = np.array([[1, 2], [3, 4]])
        >>> w = np.array([0.5, -0.5])
        >>> predict_binary(X, w, 0.0)
        array([0, 0])
    """
    # TODO: Threshold probabilities
    # proba = predict_proba_binary(X, weights, bias)
    # return (proba >= threshold).astype(int)
    raise NotImplementedError


def predict_multiclass(
    X: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """
    Make multi-class predictions: ŷ = argmax_k P(y=k|X)

    Args:
        X: Features, shape (n_samples, n_features)
        weights: Weight matrix, shape (n_features, n_classes)
        bias: Bias vector, shape (n_classes,)

    Returns:
        predictions: Class indices, shape (n_samples,)

    Example:
        >>> X = np.array([[1, 2], [3, 4]])
        >>> W = np.array([[0.5, -0.5, 0.1], [0.2, 0.3, -0.4]])
        >>> b = np.array([0.1, 0.2, 0.3])
        >>> predict_multiclass(X, W, b)
        array([0, 1])
    """
    # TODO: Get class with max probability
    # proba = predict_proba_multiclass(X, weights, bias)
    # return np.argmax(proba, axis=1)
    raise NotImplementedError
