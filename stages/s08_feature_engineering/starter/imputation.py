from __future__ import annotations

import numpy as np


def simple_imputer_strategy(X: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """
    Fill missing values using specified strategy.

    Args:
        X: Features with missing values, shape (n_samples, n_features)
        strategy: One of 'mean', 'median', 'most_frequent'

    Returns:
        X_imputed: Features with NaN filled, shape (n_samples, n_features)
    """
    raise NotImplementedError


def find_missing_mask(X: np.ndarray) -> np.ndarray:
    """
    Find where data is missing.

    Args:
        X: Features array, shape (n_samples, n_features)

    Returns:
        mask: Boolean mask, True where NaN, shape (n_samples, n_features)
    """
    raise NotImplementedError


def impute_with_constant(X: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Fill missing values with constant.

    Args:
        X: Features with missing values, shape (n_samples, n_features)
        fill_value: Value to fill NaN with

    Returns:
        X_imputed: Features with NaN filled, shape (n_samples, n_features)
    """
    raise NotImplementedError
