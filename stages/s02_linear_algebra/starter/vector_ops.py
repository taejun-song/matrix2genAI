from __future__ import annotations

import numpy as np


def vector_add(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Add two vectors element-wise.

    Args:
        v: First vector of shape (n,)
        w: Second vector of shape (n,)

    Returns:
        Sum v + w of shape (n,)

    Raises:
        ValueError: If dimensions don't match
    """
    raise NotImplementedError


def vector_scalar_multiply(v: np.ndarray, c: float) -> np.ndarray:
    """
    Multiply vector by scalar.

    Args:
        v: Vector of shape (n,)
        c: Scalar value

    Returns:
        Scaled vector c * v of shape (n,)
    """
    raise NotImplementedError


def dot_product(v: np.ndarray, w: np.ndarray) -> float:
    """
    Compute dot product of two vectors.

    Args:
        v: First vector of shape (n,)
        w: Second vector of shape (n,)

    Returns:
        Dot product v · w (scalar)

    Raises:
        ValueError: If dimensions don't match

    Hint: Use NumPy operations, no explicit loops
    """
    raise NotImplementedError


def vector_norm(v: np.ndarray, p: float = 2.0) -> float:
    """
    Compute Lp norm of a vector.

    Args:
        v: Vector of shape (n,)
        p: Norm order (1 for Manhattan, 2 for Euclidean, np.inf for max)

    Returns:
        ||v||_p = (sum(|v_i|^p))^(1/p)
        For p=inf, returns max(|v_i|)

    Hint: Handle p=np.inf as special case
    """
    raise NotImplementedError


def cosine_similarity(v: np.ndarray, w: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        v: First vector of shape (n,)
        w: Second vector of shape (n,)

    Returns:
        cos(theta) = (v · w) / (||v|| * ||w||)
        Returns 0.0 if either vector is zero

    Raises:
        ValueError: If dimensions don't match
    """
    raise NotImplementedError
