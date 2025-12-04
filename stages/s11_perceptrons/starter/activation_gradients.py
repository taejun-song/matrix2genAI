from __future__ import annotations

import numpy as np

from stages.s11_perceptrons.starter.activations import elu, softplus


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU: f'(z) = 1 if z > 0 else 0

    Args:
        z: Input values (pre-activation)

    Returns:
        gradient: Derivative at each point

    Example:
        >>> relu_derivative(np.array([-1, 0, 1]))
        array([0., 0., 1.])
    """
    # TODO: return (z > 0).astype(float)
    raise NotImplementedError


def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Derivative of Leaky ReLU: f'(z) = 1 if z > 0 else alpha

    Args:
        z: Input values
        alpha: Slope for negative region

    Returns:
        gradient: Derivative at each point

    Example:
        >>> leaky_relu_derivative(np.array([-1, 0, 1]), alpha=0.1)
        array([0.1, 0.1, 1.])
    """
    # TODO: return np.where(z > 0, 1.0, alpha)
    raise NotImplementedError


def elu_derivative(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Derivative of ELU: f'(z) = 1 if z > 0 else f(z) + alpha

    Args:
        z: Input values
        alpha: Scale parameter

    Returns:
        gradient: Derivative at each point

    Example:
        >>> elu_derivative(np.array([-1, 0, 1]), alpha=1.0)
        array([0.368, 1., 1.])
    """
    # TODO: return np.where(z > 0, 1.0, elu(z, alpha) + alpha)
    raise NotImplementedError


def gelu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of GELU (approximate).

    Args:
        z: Input values

    Returns:
        gradient: Derivative at each point

    Example:
        >>> gelu_derivative(np.array([0]))
        array([0.5])
    """
    # TODO:
    # c = np.sqrt(2 / np.pi)
    # inner = c * (z + 0.044715 * z**3)
    # tanh_inner = np.tanh(inner)
    # sech2 = 1 - tanh_inner**2
    # return 0.5 * (1 + tanh_inner) + 0.5 * z * sech2 * c * (1 + 3 * 0.044715 * z**2)
    raise NotImplementedError


def swish_derivative(z: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Derivative of Swish: f'(z) = sigmoid(βz) * (1 + βz * (1 - sigmoid(βz)))

    Args:
        z: Input values
        beta: Scaling parameter

    Returns:
        gradient: Derivative at each point

    Example:
        >>> swish_derivative(np.array([0]), beta=1.0)
        array([0.5])
    """
    # TODO:
    # sig = 1 / (1 + np.exp(-beta * z))
    # return sig * (1 + beta * z * (1 - sig))
    raise NotImplementedError


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh: f'(z) = 1 - tanh(z)^2

    Args:
        z: Input values

    Returns:
        gradient: Derivative at each point

    Example:
        >>> tanh_derivative(np.array([0]))
        array([1.])
    """
    # TODO: return 1 - np.tanh(z)**2
    raise NotImplementedError


def softplus_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of softplus: f'(z) = sigmoid(z)

    Args:
        z: Input values

    Returns:
        gradient: Derivative at each point (equals sigmoid!)

    Example:
        >>> softplus_derivative(np.array([0]))
        array([0.5])
    """
    # TODO: return 1 / (1 + np.exp(-z))
    raise NotImplementedError


def mish_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of Mish using chain rule.

    mish(z) = z * tanh(softplus(z))
    mish'(z) = tanh(sp) + z * sech^2(sp) * sp'(z)

    Args:
        z: Input values

    Returns:
        gradient: Derivative at each point

    Example:
        >>> mish_derivative(np.array([0]))
        array([0.6])
    """
    # TODO:
    # sp = softplus(z)
    # tanh_sp = np.tanh(sp)
    # sech2_sp = 1 - tanh_sp**2
    # sp_deriv = softplus_derivative(z)
    # return tanh_sp + z * sech2_sp * sp_deriv
    raise NotImplementedError
