from __future__ import annotations

from typing import Callable

import numpy as np


def gradient_descent(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float = 0.01,
    num_iters: int = 100,
) -> tuple[np.ndarray, list[float]]:
    """
    TODO: Basic gradient descent.

    Returns:
        (x_final, loss_history)
    """
    raise NotImplementedError


def gradient_descent_with_momentum(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float = 0.01,
    momentum: float = 0.9,
    num_iters: int = 100,
) -> tuple[np.ndarray, list[float]]:
    """TODO: Gradient descent with momentum."""
    raise NotImplementedError


def backtracking_line_search(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    direction: np.ndarray,
    alpha: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
) -> float:
    """TODO: Backtracking line search for step size."""
    raise NotImplementedError
