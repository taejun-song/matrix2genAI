from __future__ import annotations

import numpy as np

from stages.s03_calculus.starter.autodiff import Dual, Variable


def test_dual_addition() -> None:
    x = Dual(3.0, 1.0)
    y = Dual(4.0, 0.0)
    z = x + y
    assert np.isclose(z.value, 7.0)
    assert np.isclose(z.deriv, 1.0)


def test_dual_multiplication() -> None:
    x = Dual(3.0, 1.0)
    y = Dual(4.0, 1.0)
    z = x * y
    assert np.isclose(z.value, 12.0)
    assert np.isclose(z.deriv, 7.0)


def test_dual_sin() -> None:
    x = Dual(0.0, 1.0)
    y = x.sin()
    assert np.isclose(y.value, 0.0)
    assert np.isclose(y.deriv, 1.0)


def test_variable_backward() -> None:
    x = Variable(3.0)
    y = Variable(4.0)
    z = x * y
    z.grad = 1.0
    z.backward()
    assert np.isclose(x.grad, 4.0)
    assert np.isclose(y.grad, 3.0)
