from __future__ import annotations

import numpy as np


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
    # TODO: Implement prediction
    # HINT: Use @ operator for matrix multiplication
    # One line: return X @ weights + bias
    pass


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean squared error.

    Args:
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)

    Returns:
        mse: Mean squared error (scalar)

    Formula:
        MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²

    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 2.2, 2.8])
        >>> mse_loss(y_true, y_pred)
        0.03
    """
    # TODO: Implement MSE
    # HINT: Use np.mean() and squared differences
    # One line: return np.mean((y_true - y_pred) ** 2)
    pass


def mse_gradient(
    X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Compute gradients of MSE loss w.r.t. weights and bias.

    Args:
        X: Input features, shape (n_samples, n_features)
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)

    Returns:
        grad_w: Gradient w.r.t. weights, shape (n_features,)
        grad_b: Gradient w.r.t. bias (scalar)

    Formulas:
        errors = ŷ - y
        ∂L/∂w = (2/n) · Xᵀ · errors
        ∂L/∂b = (2/n) · Σᵢ errorsᵢ

    Example:
        >>> X = np.array([[1, 2], [3, 4]])
        >>> y_true = np.array([5, 11])
        >>> y_pred = np.array([5.5, 11.5])
        >>> grad_w, grad_b = mse_gradient(X, y_true, y_pred)
        >>> grad_w
        array([2., 3.])
        >>> grad_b
        0.5
    """
    n = len(y_true)
    errors = y_pred - y_true

    # TODO: Compute weight gradient
    # HINT: grad_w = (2/n) * X.T @ errors

    # TODO: Compute bias gradient
    # HINT: grad_b = (2/n) * np.sum(errors)

    pass


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination).

    Args:
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)

    Returns:
        r2: R² score (scalar), range (-∞, 1]

    Formula:
        SS_res = Σᵢ (yᵢ - ŷᵢ)²  (residual sum of squares)
        SS_tot = Σᵢ (yᵢ - ȳ)²   (total sum of squares)
        R² = 1 - SS_res / SS_tot

    Interpretation:
        R² = 1.0  → Perfect predictions
        R² = 0.0  → Model as good as predicting mean
        R² < 0.0  → Model worse than predicting mean

    Edge case:
        If SS_tot == 0 (all y values same), return 0.0

    Example:
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        >>> r2_score(y_true, y_pred)
        0.99
    """
    # TODO: Implement R² score
    # HINT 1: Compute SS_res = np.sum((y_true - y_pred) ** 2)
    # HINT 2: Compute y_mean and SS_tot = np.sum((y_true - y_mean) ** 2)
    # HINT 3: Handle edge case: if SS_tot == 0, return 0.0
    # HINT 4: Return 1 - SS_res / SS_tot

    pass


def normal_equation(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Solve linear regression using normal equations (closed-form).

    Args:
        X: Input features, shape (n_samples, n_features)
        y: Target values, shape (n_samples,)

    Returns:
        weights: Optimal weights, shape (n_features,)
        bias: Optimal bias (scalar)

    Formula:
        Augment X: X_aug = [1ₙ | X]
        Solve: w_aug = (X_augᵀ X_aug)⁻¹ X_augᵀ y
        Extract: bias = w_aug[0], weights = w_aug[1:]

    Example:
        >>> X = np.array([[1], [2], [3]])
        >>> y = np.array([2, 4, 6])
        >>> weights, bias = normal_equation(X, y)
        >>> weights
        array([2.])
        >>> bias
        0.0
    """
    # TODO: Implement normal equations
    # HINT 1: Create X_aug by prepending column of ones
    #         X_aug = np.column_stack([np.ones(len(X)), X])
    # HINT 2: Solve using np.linalg.lstsq (more stable than inverse)
    #         w_aug = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    # HINT 3: Extract bias (first element) and weights (rest)
    #         bias = w_aug[0]
    #         weights = w_aug[1:]

    pass


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using Z-score normalization.

    Args:
        X: Input features, shape (n_samples, n_features)

    Returns:
        X_scaled: Standardized features, shape (n_samples, n_features)
        mean: Mean of each feature, shape (n_features,)
        std: Standard deviation of each feature, shape (n_features,)

    Formula:
        X_scaled = (X - μ) / σ

    Edge case:
        If std is zero (constant feature), use std = 1 to avoid division by zero

    Example:
        >>> X = np.array([[1, 100], [2, 200], [3, 300]])
        >>> X_scaled, mean, std = standardize(X)
        >>> X_scaled
        array([[-1., -1.],
               [ 0.,  0.],
               [ 1.,  1.]])
    """
    # TODO: Implement standardization
    # HINT 1: Compute mean along axis 0: mean = np.mean(X, axis=0)
    # HINT 2: Compute std along axis 0: std = np.std(X, axis=0)
    # HINT 3: Handle zero std: std = np.where(std == 0, 1, std)
    # HINT 4: Standardize: X_scaled = (X - mean) / std

    pass


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.

    Args:
        X: Features, shape (n_samples, n_features)
        y: Targets, shape (n_samples,)
        test_size: Fraction of data for testing (default 0.2)

    Returns:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets

    Example:
        >>> X = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([1, 2, 3, 4, 5])
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> len(X_train)
        4
        >>> len(X_test)
        1
    """
    # TODO: Implement train/test split
    # HINT 1: Compute split index: split_idx = int(len(X) * (1 - test_size))
    # HINT 2: Slice arrays:
    #         X_train = X[:split_idx]
    #         X_test = X[split_idx:]
    #         y_train = y[:split_idx]
    #         y_test = y[split_idx:]

    pass


def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Generate polynomial features up to specified degree.

    Args:
        X: Input features, shape (n_samples, n_features)
        degree: Maximum polynomial degree

    Returns:
        X_poly: Polynomial features, shape (n_samples, n_poly_features)

    Example:
        >>> X = np.array([[2], [3]])
        >>> polynomial_features(X, degree=3)
        array([[ 2,  4,  8],
               [ 3,  9, 27]])

    For multiple features:
        >>> X = np.array([[2, 3]])
        >>> polynomial_features(X, degree=2)
        array([[2, 3, 4, 6, 9]])  # [x1, x2, x1², x1*x2, x2²]
    """
    # TODO: Implement polynomial feature generation
    # HINT: Use itertools.combinations_with_replacement
    #
    # from itertools import combinations_with_replacement
    #
    # n_samples, n_features = X.shape
    # features = []
    #
    # # Generate all polynomial combinations
    # for deg in range(1, degree + 1):
    #     for combo in combinations_with_replacement(range(n_features), deg):
    #         # Compute product of selected features
    #         product = X[:, combo[0]].copy()
    #         for idx in combo[1:]:
    #             product *= X[:, idx]
    #         features.append(product.reshape(-1, 1))
    #
    # return np.hstack(features)

    pass
