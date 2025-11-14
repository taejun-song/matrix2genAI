from __future__ import annotations

import numpy as np

# ============================================================================
# Building Block 1: Missing Data Detection
# ============================================================================


def find_missing_mask(X: np.ndarray) -> np.ndarray:
    """
    Find where data is missing.

    Args:
        X: Features array

    Returns:
        Boolean mask, True where NaN
    """
    # TODO: return np.isnan(X)
    raise NotImplementedError


# ============================================================================
# Building Block 2: Imputation Strategies
# ============================================================================


def _compute_fill_value(X: np.ndarray, strategy: str) -> np.ndarray:
    """
    Compute fill value per feature based on strategy.

    Args:
        X: Features (may contain NaN)
        strategy: 'mean', 'median', or 'most_frequent'

    Returns:
        Fill values per feature, shape (n_features,)
    """
    # TODO:
    # if strategy == "mean":
    #     return np.nanmean(X, axis=0)
    # elif strategy == "median":
    #     return np.nanmedian(X, axis=0)
    # elif strategy == "most_frequent":
    #     # For each column, find most common non-NaN value
    #     fill = np.zeros(X.shape[1])
    #     for i in range(X.shape[1]):
    #         col = X[:, i]
    #         col_no_nan = col[~np.isnan(col)]
    #         if len(col_no_nan) > 0:
    #             fill[i] = np.bincount(col_no_nan.astype(int)).argmax()
    #     return fill
    # else:
    #     msg = f"Unknown strategy: {strategy}"
    #     raise ValueError(msg)
    raise NotImplementedError


def simple_imputer_strategy(X: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """
    Fill missing values using strategy.

    Args:
        X: Features with missing values
        strategy: 'mean', 'median', or 'most_frequent'

    Returns:
        X_imputed: Features with NaN filled
    """
    # TODO:
    # X_imputed = X.copy()
    # fill_values = _compute_fill_value(X, strategy)
    # mask = find_missing_mask(X)
    # for i in range(X.shape[1]):
    #     X_imputed[mask[:, i], i] = fill_values[i]
    # return X_imputed
    raise NotImplementedError


def impute_with_constant(X: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Fill missing values with constant.

    Args:
        X: Features with missing values
        fill_value: Value to fill NaN with

    Returns:
        X_imputed: Features with NaN filled
    """
    # TODO:
    # X_imputed = X.copy()
    # X_imputed[find_missing_mask(X)] = fill_value
    # return X_imputed
    raise NotImplementedError
