from __future__ import annotations

import numpy as np

from stages.s05_optimization.starter.gradient_descent import (
    gradient_descent,
    gradient_descent_with_momentum,
)


def test_gradient_descent_quadratic() -> None:
    f = lambda x: np.sum(x**2)
    grad_f = lambda x: 2*x
    x0 = np.array([5.0, 5.0])
    x_final, loss_history = gradient_descent(f, grad_f, x0, lr=0.1, num_iters=100)
    assert np.allclose(x_final, np.zeros(2), atol=1e-4)
    assert loss_history[-1] < loss_history[0]


def test_gradient_descent_with_momentum_converges() -> None:
    f = lambda x: np.sum(x**2)
    grad_f = lambda x: 2*x
    x0 = np.array([5.0, 5.0])
    x_final, loss_history = gradient_descent_with_momentum(
        f, grad_f, x0, lr=0.1, momentum=0.9, num_iters=100
    )
    assert np.allclose(x_final, np.zeros(2), atol=1e-4)
