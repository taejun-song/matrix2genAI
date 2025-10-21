from __future__ import annotations

import numpy as np

from stages.s05_optimization.starter.optimizers import adam, adagrad, rmsprop


def test_adagrad_converges() -> None:
    f = lambda x: np.sum(x**2)
    grad_f = lambda x: 2*x
    x0 = np.array([5.0, 5.0])
    x_final, _ = adagrad(f, grad_f, x0, lr=0.5, num_iters=100)
    assert np.allclose(x_final, np.zeros(2), atol=1e-2)


def test_rmsprop_converges() -> None:
    f = lambda x: np.sum(x**2)
    grad_f = lambda x: 2*x
    x0 = np.array([5.0, 5.0])
    x_final, _ = rmsprop(f, grad_f, x0, lr=0.1, num_iters=100)
    assert np.allclose(x_final, np.zeros(2), atol=1e-3)


def test_adam_converges() -> None:
    f = lambda x: np.sum(x**2)
    grad_f = lambda x: 2*x
    x0 = np.array([5.0, 5.0])
    x_final, _ = adam(f, grad_f, x0, lr=0.1, num_iters=100)
    assert np.allclose(x_final, np.zeros(2), atol=1e-3)
