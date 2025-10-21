from __future__ import annotations

import numpy as np


def mean(x: np.ndarray) -> float:
    """TODO: Compute sample mean."""
    raise NotImplementedError


def variance(x: np.ndarray, ddof: int = 0) -> float:
    """TODO: Compute sample variance."""
    raise NotImplementedError


def covariance(x: np.ndarray, y: np.ndarray) -> float:
    """TODO: Compute covariance between x and y."""
    raise NotImplementedError


def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """TODO: Compute covariance matrix for data matrix X (n_samples x n_features)."""
    raise NotImplementedError
