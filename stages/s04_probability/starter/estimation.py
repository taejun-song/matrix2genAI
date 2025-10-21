from __future__ import annotations

import numpy as np


def mle_normal(x: np.ndarray) -> tuple[float, float]:
    """TODO: MLE for normal distribution. Return (mean, std)."""
    raise NotImplementedError


def mle_bernoulli(x: np.ndarray) -> float:
    """TODO: MLE for Bernoulli parameter p."""
    raise NotImplementedError


def map_normal(x: np.ndarray, prior_mean: float, prior_std: float) -> float:
    """TODO: MAP estimation for normal distribution mean with Gaussian prior."""
    raise NotImplementedError
