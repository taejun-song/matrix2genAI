from __future__ import annotations


class Dual:
    """Dual number for forward-mode automatic differentiation."""

    def __init__(self, value: float, deriv: float = 0.0):
        self.value = value
        self.deriv = deriv

    def __add__(self, other: Dual | float) -> Dual:
        """TODO: Implement addition."""
        raise NotImplementedError

    def __mul__(self, other: Dual | float) -> Dual:
        """TODO: Implement multiplication."""
        raise NotImplementedError

    def __sub__(self, other: Dual | float) -> Dual:
        """TODO: Implement subtraction."""
        raise NotImplementedError

    def __truediv__(self, other: Dual | float) -> Dual:
        """TODO: Implement division."""
        raise NotImplementedError

    def __pow__(self, n: float) -> Dual:
        """TODO: Implement power."""
        raise NotImplementedError

    def sin(self) -> Dual:
        """TODO: Implement sin."""
        raise NotImplementedError

    def cos(self) -> Dual:
        """TODO: Implement cos."""
        raise NotImplementedError

    def exp(self) -> Dual:
        """TODO: Implement exp."""
        raise NotImplementedError


class Variable:
    """Computational graph node for reverse-mode AD."""

    def __init__(self, value: float):
        self.value = value
        self.grad = 0.0
        self.backward_fn = lambda: None

    def backward(self) -> None:
        """TODO: Run backpropagation from this node."""
        raise NotImplementedError

    def __add__(self, other: Variable | float) -> Variable:
        """TODO: Implement addition with backprop."""
        raise NotImplementedError

    def __mul__(self, other: Variable | float) -> Variable:
        """TODO: Implement multiplication with backprop."""
        raise NotImplementedError
