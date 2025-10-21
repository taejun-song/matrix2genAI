from __future__ import annotations

from typing import Callable, Literal

import numpy as np

Method = Literal["forward", "central", "backward"]


def finite_difference(
    f: Callable[[float], float], x: float, h: float = 1e-5, method: Method = "central"
) -> float:
    """TODO: Compute derivative of scalar function using finite differences."""
    raise NotImplementedError


def gradient_finite_diff(
    f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-5
) -> np.ndarray:
    """TODO: Compute gradient using finite differences."""
    raise NotImplementedError


def jacobian_finite_diff(
    f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, h: float = 1e-5
) -> np.ndarray:
    """TODO: Compute Jacobian matrix."""
    raise NotImplementedError


def hessian_finite_diff(
    f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-5
) -> np.ndarray:
    """TODO: Compute Hessian matrix (second derivatives)."""
    raise NotImplementedError


def gradient_check(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    epsilon: float = 1e-5,
    tolerance: float = 1e-7,
) -> bool:
    """TODO: Check analytical gradient against numerical gradient."""
    raise NotImplementedError
