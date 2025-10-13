from __future__ import annotations

from collections.abc import Callable

import numpy as np


def finite_diff_grad(
    f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Central finite-difference gradient for scalar f."""
    g = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps
        g.flat[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


def relative_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(a) + np.linalg.norm(b) + eps
    return float(num / den), float(num)
