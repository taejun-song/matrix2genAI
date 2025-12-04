from __future__ import annotations

import numpy as np


def xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """
    Xavier/Glorot uniform initialization.

    Args:
        fan_in: Number of input units
        fan_out: Number of output units

    Returns:
        weights: Shape (fan_in, fan_out), uniform in [-limit, limit]
                 where limit = sqrt(6 / (fan_in + fan_out))
    """
    # TODO:
    # limit = np.sqrt(6 / (fan_in + fan_out))
    # return np.random.uniform(-limit, limit, (fan_in, fan_out))
    raise NotImplementedError


def xavier_normal(fan_in: int, fan_out: int) -> np.ndarray:
    """
    Xavier/Glorot normal initialization.

    Args:
        fan_in: Number of input units
        fan_out: Number of output units

    Returns:
        weights: Shape (fan_in, fan_out), normal with std = sqrt(2 / (fan_in + fan_out))
    """
    # TODO:
    # std = np.sqrt(2 / (fan_in + fan_out))
    # return np.random.randn(fan_in, fan_out) * std
    raise NotImplementedError


def he_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """
    He uniform initialization (for ReLU networks).

    Args:
        fan_in: Number of input units
        fan_out: Number of output units

    Returns:
        weights: Shape (fan_in, fan_out), uniform in [-limit, limit]
                 where limit = sqrt(6 / fan_in)
    """
    # TODO:
    # limit = np.sqrt(6 / fan_in)
    # return np.random.uniform(-limit, limit, (fan_in, fan_out))
    raise NotImplementedError


def he_normal(fan_in: int, fan_out: int) -> np.ndarray:
    """
    He normal initialization (for ReLU networks).

    Args:
        fan_in: Number of input units
        fan_out: Number of output units

    Returns:
        weights: Shape (fan_in, fan_out), normal with std = sqrt(2 / fan_in)
    """
    # TODO:
    # std = np.sqrt(2 / fan_in)
    # return np.random.randn(fan_in, fan_out) * std
    raise NotImplementedError


def zeros(shape: tuple[int, ...]) -> np.ndarray:
    """
    Zero initialization (typically for biases).

    Args:
        shape: Shape of the array

    Returns:
        zeros: Array of zeros
    """
    # TODO: return np.zeros(shape)
    raise NotImplementedError
