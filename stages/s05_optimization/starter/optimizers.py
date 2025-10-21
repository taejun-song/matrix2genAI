from __future__ import annotations

from typing import Callable

import numpy as np


def adagrad(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float = 0.01,
    num_iters: int = 100,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, list[float]]:
    """TODO: AdaGrad optimizer."""
    raise NotImplementedError


def rmsprop(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float = 0.001,
    beta: float = 0.9,
    num_iters: int = 100,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, list[float]]:
    """TODO: RMSProp optimizer."""
    raise NotImplementedError


def adam(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    num_iters: int = 100,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, list[float]]:
    """TODO: Adam optimizer."""
    raise NotImplementedError
