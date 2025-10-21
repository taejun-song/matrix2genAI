from __future__ import annotations

import numpy as np


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Lx = b for lower triangular matrix L.

    Args:
        L: Lower triangular matrix of shape (n, n)
        b: Right-hand side vector of shape (n,)

    Returns:
        Solution vector x of shape (n,)

    Raises:
        ValueError: If L is singular (zero diagonal element)

    Hint: x[i] = (b[i] - sum(L[i,j]*x[j] for j<i)) / L[i,i]
    """
    raise NotImplementedError


def backward_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ux = b for upper triangular matrix U.

    Args:
        U: Upper triangular matrix of shape (n, n)
        b: Right-hand side vector of shape (n,)

    Returns:
        Solution vector x of shape (n,)

    Raises:
        ValueError: If U is singular (zero diagonal element)

    Hint: Similar to forward substitution but iterate backwards
    """
    raise NotImplementedError


def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.

    Args:
        A: Coefficient matrix of shape (n, n)
        b: Right-hand side vector of shape (n,)

    Returns:
        Solution vector x of shape (n,)

    Raises:
        ValueError: If system is singular (no unique solution)

    Hint:
    1. Create augmented matrix [A | b]
    2. Forward elimination with partial pivoting (swap rows to get largest pivot)
    3. Backward substitution to solve
    4. Check for zero pivots (singular system)
    """
    raise NotImplementedError


def lu_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose A = LU using Doolittle algorithm.

    Args:
        A: Square matrix of shape (n, n)

    Returns:
        (L, U) where:
        - L is lower triangular with ones on diagonal, shape (n, n)
        - U is upper triangular, shape (n, n)

    Raises:
        ValueError: If decomposition fails (matrix is singular)

    Hint: Doolittle algorithm
    - L has ones on diagonal
    - For each column k:
      - Compute U[k, k:] (upper part)
      - Compute L[k+1:, k] (lower part)
    """
    raise NotImplementedError
