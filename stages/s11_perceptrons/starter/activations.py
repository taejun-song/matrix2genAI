from __future__ import annotations

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """
    ReLU activation: f(z) = max(0, z)

    Args:
        z: Input values, any shape

    Returns:
        activated: ReLU applied element-wise

    Example:
        >>> relu(np.array([-2, -1, 0, 1, 2]))
        array([0, 0, 0, 1, 2])
    """
    # TODO: return np.maximum(0, z)
    raise NotImplementedError


def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky ReLU: f(z) = z if z > 0 else alpha * z

    Args:
        z: Input values
        alpha: Slope for negative region (default 0.01)

    Returns:
        activated: Leaky ReLU applied element-wise

    Example:
        >>> leaky_relu(np.array([-2, 0, 2]), alpha=0.1)
        array([-0.2, 0, 2])
    """
    # TODO: return np.where(z > 0, z, alpha * z)
    raise NotImplementedError


def elu(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    ELU: f(z) = z if z > 0 else alpha * (exp(z) - 1)

    Args:
        z: Input values
        alpha: Scale for negative region (default 1.0)

    Returns:
        activated: ELU applied element-wise

    Example:
        >>> elu(np.array([-1, 0, 1]), alpha=1.0)
        array([-0.632, 0, 1])
    """
    # TODO: return np.where(z > 0, z, alpha * (np.exp(z) - 1))
    raise NotImplementedError


def gelu(z: np.ndarray) -> np.ndarray:
    """
    GELU: f(z) ≈ 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))

    Used in transformers (BERT, GPT).

    Args:
        z: Input values

    Returns:
        activated: GELU applied element-wise

    Example:
        >>> gelu(np.array([-1, 0, 1]))
        array([-0.159, 0, 0.841])
    """
    # TODO:
    # c = np.sqrt(2 / np.pi)
    # return 0.5 * z * (1 + np.tanh(c * (z + 0.044715 * z**3)))
    raise NotImplementedError


def swish(z: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Swish / SiLU: f(z) = z * sigmoid(beta * z)

    Args:
        z: Input values
        beta: Scaling parameter (default 1.0)

    Returns:
        activated: Swish applied element-wise

    Example:
        >>> swish(np.array([-1, 0, 1]), beta=1.0)
        array([-0.269, 0, 0.731])
    """
    # TODO:
    # sigmoid = 1 / (1 + np.exp(-beta * z))
    # return z * sigmoid
    raise NotImplementedError


def tanh_activation(z: np.ndarray) -> np.ndarray:
    """
    Tanh activation: f(z) = (e^z - e^-z) / (e^z + e^-z)

    Args:
        z: Input values

    Returns:
        activated: Tanh applied element-wise, values in (-1, 1)

    Example:
        >>> tanh_activation(np.array([-1, 0, 1]))
        array([-0.762, 0, 0.762])
    """
    # TODO: return np.tanh(z)
    raise NotImplementedError


def softplus(z: np.ndarray) -> np.ndarray:
    """
    Softplus: f(z) = log(1 + exp(z))

    Smooth approximation of ReLU.

    Args:
        z: Input values

    Returns:
        activated: Softplus applied element-wise

    Example:
        >>> softplus(np.array([-1, 0, 1]))
        array([0.313, 0.693, 1.313])
    """
    # TODO: Use numerically stable version
    # return np.where(z > 20, z, np.log1p(np.exp(z)))
    raise NotImplementedError


def mish(z: np.ndarray) -> np.ndarray:
    """
    Mish: f(z) = z * tanh(softplus(z))

    Self-regularized activation function.

    Args:
        z: Input values

    Returns:
        activated: Mish applied element-wise

    Example:
        >>> mish(np.array([-1, 0, 1]))
        array([-0.303, 0, 0.865])
    """
    # TODO: return z * np.tanh(softplus(z))
    raise NotImplementedError
