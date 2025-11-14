from __future__ import annotations

import numpy as np

# ============================================================================
# Building Block 1: Linear Forward Pass
# ============================================================================


def predict(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Compute linear predictions: ŷ = Xw + b

    Args:
        X: Input features, shape (n_samples, n_features)
        weights: Weight vector, shape (n_features,)
        bias: Bias term (scalar)

    Returns:
        predictions: Predicted values, shape (n_samples,)

    Example:
        >>> X = np.array([[1, 2], [3, 4]])
        >>> w = np.array([0.5, 1.0])
        >>> b = 0.1
        >>> predict(X, w, b)
        array([2.6, 5.6])
    """
    # TODO: return X @ weights + bias
    raise NotImplementedError


# ============================================================================
# Building Block 2: Loss Function Components
# ============================================================================


def _squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute squared errors per sample: (y - ŷ)²

    Args:
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)

    Returns:
        Squared errors, shape (n_samples,)

    Example:
        >>> _squared_error(np.array([1, 2, 3]), np.array([1.5, 2.0, 2.5]))
        array([0.25, 0.0, 0.25])
    """
    # TODO: return (y_true - y_pred) ** 2
    raise NotImplementedError


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean squared error: MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²

    Args:
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)

    Returns:
        mse: Mean squared error (scalar)

    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 2.2, 2.8])
        >>> mse_loss(y_true, y_pred)
        0.03
    """
    # TODO: return float(np.mean(_squared_error(y_true, y_pred)))
    raise NotImplementedError


# ============================================================================
# Building Block 3: Gradient Components
# ============================================================================


def _prediction_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute prediction errors: error = ŷ - y

    Args:
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)

    Returns:
        Prediction errors, shape (n_samples,)

    Example:
        >>> _prediction_error(np.array([1, 2, 3]), np.array([1.5, 2.0, 2.5]))
        array([0.5, 0.0, -0.5])
    """
    # TODO: return y_pred - y_true
    raise NotImplementedError


def mse_gradient(
    X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Compute gradients: ∂L/∂w = (2/n)X^T·error, ∂L/∂b = (2/n)Σerror

    Args:
        X: Input features, shape (n_samples, n_features)
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)

    Returns:
        grad_w: Gradient w.r.t. weights, shape (n_features,)
        grad_b: Gradient w.r.t. bias (scalar)

    Example:
        >>> X = np.array([[1, 2], [3, 4]])
        >>> y_true = np.array([5, 11])
        >>> y_pred = np.array([5.5, 11.5])
        >>> grad_w, grad_b = mse_gradient(X, y_true, y_pred)
        >>> grad_w
        array([2., 3.])
    """
    # TODO:
    # n = len(y_true)
    # error = _prediction_error(y_true, y_pred)
    # grad_w = (2 / n) * (X.T @ error)
    # grad_b = (2 / n) * np.sum(error)
    # return grad_w, float(grad_b)
    raise NotImplementedError


# ============================================================================
# Building Block 4: R² Score Components
# ============================================================================


def _sum_of_squared_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute SS_res = Σᵢ (yᵢ - ŷᵢ)²

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Sum of squared residuals

    Example:
        >>> _sum_of_squared_residuals(np.array([1, 2, 3]), np.array([1, 2, 2.5]))
        0.25
    """
    # TODO: return float(np.sum((y_true - y_pred) ** 2))
    raise NotImplementedError


def _total_sum_of_squares(y_true: np.ndarray) -> float:
    """
    Compute SS_tot = Σᵢ (yᵢ - ȳ)²

    Args:
        y_true: True values

    Returns:
        Total sum of squares

    Example:
        >>> _total_sum_of_squares(np.array([1, 2, 3, 4, 5]))
        10.0
    """
    # TODO: return float(np.sum((y_true - np.mean(y_true)) ** 2))
    raise NotImplementedError


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² = 1 - SS_res / SS_tot

    Args:
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)

    Returns:
        r2: R² score (scalar)

    Example:
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        >>> r2_score(y_true, y_pred)
        0.99
    """
    # TODO:
    # ss_res = _sum_of_squared_residuals(y_true, y_pred)
    # ss_tot = _total_sum_of_squares(y_true)
    # if ss_tot == 0:
    #     return 0.0
    # return 1 - ss_res / ss_tot
    raise NotImplementedError


# ============================================================================
# Building Block 5: Normal Equation Components
# ============================================================================


def _augment_with_bias(X: np.ndarray) -> np.ndarray:
    """
    Prepend column of ones to X: X_aug = [1ₙ | X]

    Args:
        X: Features, shape (n_samples, n_features)

    Returns:
        X_aug: Augmented features, shape (n_samples, n_features+1)

    Example:
        >>> _augment_with_bias(np.array([[1, 2], [3, 4]]))
        array([[1, 1, 2], [1, 3, 4]])
    """
    # TODO: return np.column_stack([np.ones(len(X)), X])
    raise NotImplementedError


def normal_equation(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Solve linear regression: w_aug = (X^T X)^(-1) X^T y

    Args:
        X: Input features, shape (n_samples, n_features)
        y: Target values, shape (n_samples,)

    Returns:
        weights: Optimal weights, shape (n_features,)
        bias: Optimal bias (scalar)

    Example:
        >>> X = np.array([[1], [2], [3]])
        >>> y = np.array([2, 4, 6])
        >>> weights, bias = normal_equation(X, y)
        >>> weights
        array([2.])
    """
    # TODO:
    # X_aug = _augment_with_bias(X)
    # w_aug = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    # bias = w_aug[0]
    # weights = w_aug[1:]
    # return weights, float(bias)
    raise NotImplementedError


# ============================================================================
# Building Block 6: Standardization Components
# ============================================================================


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Divide, replacing division by zero with zero.

    Args:
        numerator: Numerator array
        denominator: Denominator array

    Returns:
        Result with safe division

    Example:
        >>> _safe_divide(np.array([1, 2]), np.array([2, 0]))
        array([0.5, 0.0])
    """
    # TODO: return np.where(denominator == 0, 0, numerator / denominator)
    raise NotImplementedError


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize: X_scaled = (X - μ) / σ

    Args:
        X: Input features, shape (n_samples, n_features)

    Returns:
        X_scaled: Standardized features
        mean: Mean of each feature
        std: Standard deviation of each feature

    Example:
        >>> X = np.array([[1, 100], [2, 200], [3, 300]])
        >>> X_scaled, mean, std = standardize(X)
        >>> X_scaled
        array([[-1., -1.], [0., 0.], [1., 1.]])
    """
    # TODO:
    # mean = np.mean(X, axis=0)
    # std = np.std(X, axis=0)
    # std = np.where(std == 0, 1, std)
    # X_scaled = (X - mean) / std
    # return X_scaled, mean, std
    raise NotImplementedError


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/test sets.

    Args:
        X: Features
        y: Targets
        test_size: Fraction for testing

    Returns:
        X_train, X_test, y_train, y_test

    Example:
        >>> X = np.arange(10).reshape(10, 1)
        >>> y = np.arange(10)
        >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
        >>> len(X_tr), len(X_te)
        (8, 2)
    """
    # TODO:
    # split_idx = int(len(X) * (1 - test_size))
    # return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    raise NotImplementedError


def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Generate polynomial features.

    Args:
        X: Input features
        degree: Maximum polynomial degree

    Returns:
        X_poly: Polynomial features

    Example:
        >>> X = np.array([[2], [3]])
        >>> polynomial_features(X, degree=3)
        array([[2, 4, 8], [3, 9, 27]])
    """
    # TODO: Use itertools.combinations_with_replacement
    # from itertools import combinations_with_replacement
    # features = []
    # for deg in range(1, degree + 1):
    #     for combo in combinations_with_replacement(range(X.shape[1]), deg):
    #         product = X[:, combo[0]].copy()
    #         for idx in combo[1:]:
    #             product *= X[:, idx]
    #         features.append(product.reshape(-1, 1))
    # return np.hstack(features)
    raise NotImplementedError
