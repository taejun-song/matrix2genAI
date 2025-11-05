from __future__ import annotations

import numpy as np


def min_max_scale(
    X: np.ndarray, feature_range: tuple[float, float] = (0, 1)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale features to given range.

    Args:
        X: Features, shape (n_samples, n_features)
        feature_range: Target range (min, max)

    Returns:
        X_scaled: Scaled features, shape (n_samples, n_features)
        X_min: Min per feature, shape (n_features,)
        X_max: Max per feature, shape (n_features,)
    """
    raise NotImplementedError


def robust_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale features using median and IQR.

    Args:
        X: Features, shape (n_samples, n_features)

    Returns:
        X_scaled: Scaled features, shape (n_samples, n_features)
        median: Median per feature, shape (n_features,)
        iqr: IQR per feature, shape (n_features,)
    """
    raise NotImplementedError


def variance_threshold_select(X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Select features with variance above threshold.

    Args:
        X: Features, shape (n_samples, n_features)
        threshold: Minimum variance

    Returns:
        selected_features: Boolean mask, shape (n_features,)
    """
    raise NotImplementedError


def correlation_filter(X: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """
    Remove highly correlated features.

    Args:
        X: Features, shape (n_samples, n_features)
        threshold: Maximum absolute correlation

    Returns:
        selected_features: Boolean mask, shape (n_features,)
    """
    raise NotImplementedError


def detect_outliers_iqr(X: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """
    Detect outliers using IQR method.

    Args:
        X: Features, shape (n_samples, n_features)
        multiplier: IQR multiplier (typically 1.5)

    Returns:
        outlier_mask: Boolean, True for outliers, shape (n_samples, n_features)
    """
    raise NotImplementedError
