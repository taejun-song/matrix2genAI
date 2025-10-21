from __future__ import annotations

import numpy as np


def matrix_multiply_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices using explicit triple loop (naive O(n^3) algorithm).

    Args:
        A: Matrix of shape (m, k)
        B: Matrix of shape (k, n)

    Returns:
        Product AB of shape (m, n)

    Raises:
        ValueError: If dimensions incompatible (A columns != B rows)

    Hint: Use three nested loops: for i, for j, for k
    """
    raise NotImplementedError


def matrix_multiply_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices using NumPy vectorization.

    Args:
        A: Matrix of shape (m, k)
        B: Matrix of shape (k, n)

    Returns:
        Product AB of shape (m, n)

    Raises:
        ValueError: If dimensions incompatible

    Hint: Use np.dot or @ operator, or broadcasting with np.sum
    """
    raise NotImplementedError


def matrix_transpose(A: np.ndarray) -> np.ndarray:
    """
    Transpose a matrix.

    Args:
        A: Matrix of shape (m, n)

    Returns:
        Transpose A^T of shape (n, m)
    """
    raise NotImplementedError


def matrix_trace(A: np.ndarray) -> float:
    """
    Compute trace of a square matrix.

    Args:
        A: Square matrix of shape (n, n)

    Returns:
        Trace tr(A) = sum of diagonal elements

    Raises:
        ValueError: If matrix is not square
    """
    raise NotImplementedError


def frobenius_norm(A: np.ndarray) -> float:
    """
    Compute Frobenius norm of a matrix.

    Args:
        A: Matrix of shape (m, n)

    Returns:
        ||A||_F = sqrt(sum(A_ij^2))

    Hint: Flatten the matrix or use element-wise operations
    """
    raise NotImplementedError
