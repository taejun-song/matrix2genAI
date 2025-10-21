from __future__ import annotations

import numpy as np
import pytest

from stages.s03_calculus.starter.numerical_diff import (
    finite_difference,
    gradient_check,
    gradient_finite_diff,
    hessian_finite_diff,
    jacobian_finite_diff,
)


def test_finite_difference_quadratic() -> None:
    f = lambda x: x**2
    deriv = finite_difference(f, 3.0)
    assert np.isclose(deriv, 6.0, atol=1e-5)


def test_finite_difference_sin() -> None:
    f = np.sin
    deriv = finite_difference(f, 0.0)
    assert np.isclose(deriv, 1.0, atol=1e-6)


def test_gradient_finite_diff_simple() -> None:
    f = lambda x: x[0]**2 + x[1]**2
    x = np.array([1.0, 2.0])
    grad = gradient_finite_diff(f, x)
    expected = np.array([2.0, 4.0])
    assert np.allclose(grad, expected, atol=1e-5)


def test_jacobian_finite_diff() -> None:
    f = lambda x: np.array([x[0]**2, x[0]*x[1]])
    x = np.array([2.0, 3.0])
    jac = jacobian_finite_diff(f, x)
    expected = np.array([[4.0, 0.0], [3.0, 2.0]])
    assert np.allclose(jac, expected, atol=1e-5)


def test_hessian_finite_diff() -> None:
    f = lambda x: x[0]**2 + x[1]**2
    x = np.array([1.0, 1.0])
    hess = hessian_finite_diff(f, x)
    expected = np.array([[2.0, 0.0], [0.0, 2.0]])
    assert np.allclose(hess, expected, atol=1e-4)


def test_gradient_check_passes() -> None:
    f = lambda x: np.sum(x**2)
    grad_f = lambda x: 2*x
    x = np.random.randn(5)
    assert gradient_check(f, grad_f, x)
