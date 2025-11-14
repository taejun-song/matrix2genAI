from __future__ import annotations

import numpy as np

# ============================================================================
# Building Block 1: Min-Max Scaling Components
# ============================================================================


def _compute_range(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute min and max per feature.

    Args:
        X: Features, shape (n_samples, n_features)

    Returns:
        X_min: Min per feature, shape (n_features,)
        X_max: Max per feature, shape (n_features,)
    """
    # TODO: return np.min(X, axis=0), np.max(X, axis=0)
    raise NotImplementedError


def min_max_scale(
    X: np.ndarray, feature_range: tuple[float, float] = (0, 1)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale to [min_val, max_val]: X_scaled = (X-X_min)/(X_max-X_min)*(max-min)+min

    Args:
        X: Features, shape (n_samples, n_features)
        feature_range: Target range (min, max)

    Returns:
        X_scaled, X_min, X_max
    """
    # TODO:
    # X_min, X_max = _compute_range(X)
    # X_std = (X - X_min) / np.where(X_max - X_min == 0, 1, X_max - X_min)
    # min_val, max_val = feature_range
    # X_scaled = X_std * (max_val - min_val) + min_val
    # return X_scaled, X_min, X_max
    raise NotImplementedError


# ============================================================================
# Building Block 2: Robust Scaling Components
# ============================================================================


def _compute_iqr(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute IQR = Q3 - Q1 per feature.

    Args:
        X: Features

    Returns:
        q1: 25th percentile per feature
        q3: 75th percentile per feature
    """
    # TODO:
    # q1 = np.percentile(X, 25, axis=0)
    # q3 = np.percentile(X, 75, axis=0)
    # return q1, q3
    raise NotImplementedError


def robust_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale using median and IQR: X_scaled = (X - median) / IQR

    Args:
        X: Features

    Returns:
        X_scaled, median, iqr
    """
    # TODO:
    # median = np.median(X, axis=0)
    # q1, q3 = _compute_iqr(X)
    # iqr = q3 - q1
    # iqr = np.where(iqr == 0, 1, iqr)
    # X_scaled = (X - median) / iqr
    # return X_scaled, median, iqr
    raise NotImplementedError


# ============================================================================
# Building Block 3: Feature Selection Components
# ============================================================================


def variance_threshold_select(X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Select features with variance > threshold.

    Args:
        X: Features
        threshold: Minimum variance

    Returns:
        Boolean mask, shape (n_features,)
    """
    # TODO: return np.var(X, axis=0) > threshold
    raise NotImplementedError


def _correlation_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix.

    Args:
        X: Features

    Returns:
        Correlation matrix, shape (n_features, n_features)
    """
    # TODO: return np.corrcoef(X, rowvar=False)
    raise NotImplementedError


def correlation_filter(X: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """
    Remove highly correlated features (keep first of pair).

    Args:
        X: Features
        threshold: Max absolute correlation

    Returns:
        Boolean mask
    """
    # TODO:
    # corr_matrix = _correlation_matrix(X)
    # selected = np.ones(X.shape[1], dtype=bool)
    # for i in range(len(selected)):
    #     for j in range(i + 1, len(selected)):
    #         if abs(corr_matrix[i, j]) > threshold:
    #             selected[j] = False
    # return selected
    raise NotImplementedError


# ============================================================================
# Building Block 4: Outlier Detection Components
# ============================================================================


def detect_outliers_iqr(X: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """
    Detect outliers: value < Q1 - multiplier*IQR or value > Q3 + multiplier*IQR

    Args:
        X: Features
        multiplier: IQR multiplier

    Returns:
        Boolean mask, True for outliers
    """
    # TODO:
    # q1, q3 = _compute_iqr(X)
    # iqr = q3 - q1
    # lower_bound = q1 - multiplier * iqr
    # upper_bound = q3 + multiplier * iqr
    # return (X < lower_bound) | (X > upper_bound)
    raise NotImplementedError
