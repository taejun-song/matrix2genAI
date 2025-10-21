from __future__ import annotations

import numpy as np


def normal_pdf(x: np.ndarray, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """TODO: Compute normal/Gaussian PDF."""
    raise NotImplementedError


def normal_sample(mean: float = 0.0, std: float = 1.0, size: int = 1) -> np.ndarray:
    """TODO: Sample from normal distribution using Box-Muller transform."""
    raise NotImplementedError


def bernoulli_pmf(k: np.ndarray, p: float) -> np.ndarray:
    """TODO: Compute Bernoulli PMF."""
    raise NotImplementedError


def bernoulli_sample(p: float, size: int = 1) -> np.ndarray:
    """TODO: Sample from Bernoulli distribution."""
    raise NotImplementedError


def categorical_pmf(k: int, probs: np.ndarray) -> float:
    """TODO: Compute categorical PMF."""
    raise NotImplementedError


def categorical_sample(probs: np.ndarray, size: int = 1) -> np.ndarray:
    """TODO: Sample from categorical distribution."""
    raise NotImplementedError
