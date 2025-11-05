from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from stages.s05_optimization.starter.optimizers import adagrad, adam, rmsprop


def test_adagrad_converges() -> None:
    def f(x: NDArray[np.floating]) -> float:
        return float(np.sum(x**2))
    def grad_f(x: NDArray[np.floating]) -> NDArray[np.floating]:
        return 2*x
    x0 = np.array([5.0, 5.0])
    x_final, _ = adagrad(f, grad_f, x0, lr=0.5, num_iters=100)
    assert np.allclose(x_final, np.zeros(2), atol=1e-2)


def test_rmsprop_converges() -> None:
    def f(x: NDArray[np.floating]) -> float:
        return float(np.sum(x**2))
    def grad_f(x: NDArray[np.floating]) -> NDArray[np.floating]:
        return 2*x
    x0 = np.array([5.0, 5.0])
    x_final, _ = rmsprop(f, grad_f, x0, lr=0.1, num_iters=100)
    assert np.allclose(x_final, np.zeros(2), atol=1e-3)


def test_adam_converges() -> None:
    def f(x: NDArray[np.floating]) -> float:
        return float(np.sum(x**2))
    def grad_f(x: NDArray[np.floating]) -> NDArray[np.floating]:
        return 2*x
    x0 = np.array([5.0, 5.0])
    x_final, _ = adam(f, grad_f, x0, lr=0.1, num_iters=100)
    assert np.allclose(x_final, np.zeros(2), atol=1e-3)
