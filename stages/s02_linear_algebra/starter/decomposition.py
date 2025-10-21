from __future__ import annotations

import numpy as np


def qr_decomposition_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose A = QR using Gram-Schmidt orthogonalization.

    Args:
        A: Matrix of shape (m, n) with linearly independent columns

    Returns:
        (Q, R) where:
        - Q is orthogonal matrix (Q^T Q = I), shape (m, n)
        - R is upper triangular matrix, shape (n, n)

    Raises:
        ValueError: If A has linearly dependent columns

    Hint: Modified Gram-Schmidt for better numerical stability
    1. For each column a_i of A:
       - Start with u_i = a_i
       - Subtract projections onto previous q_j: u_i -= (q_j^T a_i) q_j
       - Normalize: q_i = u_i / ||u_i||
       - Store R[j,i] = q_j^T a_i
    """
    raise NotImplementedError


def power_iteration(
    A: np.ndarray, num_iters: int = 100, tol: float = 1e-6
) -> tuple[float, np.ndarray]:
    """
    Find dominant eigenvalue and eigenvector using power iteration.

    Args:
        A: Square matrix of shape (n, n)
        num_iters: Maximum number of iterations
        tol: Convergence tolerance for eigenvalue

    Returns:
        (eigenvalue, eigenvector) where:
        - eigenvalue: Dominant eigenvalue (largest magnitude)
        - eigenvector: Corresponding eigenvector, normalized to ||v|| = 1

    Hint: Power iteration algorithm
    1. Start with random vector v
    2. Repeat:
       - v_new = A @ v
       - v_new = v_new / ||v_new||  (normalize)
       - eigenvalue = v^T A v  (Rayleigh quotient)
       - Check convergence: if change in eigenvalue < tol, stop
    3. Return eigenvalue and eigenvector
    """
    raise NotImplementedError


def determinant_recursive(A: np.ndarray) -> float:
    """
    Compute determinant using cofactor expansion (recursive).

    Args:
        A: Square matrix of shape (n, n), preferably small (n <= 4)

    Returns:
        det(A)

    Hint: Cofactor expansion along first row
    - Base case: 1x1 matrix, return A[0,0]
    - Base case: 2x2 matrix, return A[0,0]*A[1,1] - A[0,1]*A[1,0]
    - Recursive: det(A) = sum((-1)^j * A[0,j] * det(minor(A, 0, j)))
      where minor(A, i, j) is A with row i and column j removed

    Warning: O(n!) complexity, only use for small matrices
    """
    raise NotImplementedError


def determinant_lu(A: np.ndarray) -> float:
    """
    Compute determinant using LU decomposition.

    Args:
        A: Square matrix of shape (n, n)

    Returns:
        det(A)

    Hint:
    - det(A) = det(L) * det(U)
    - For triangular matrices, det = product of diagonal elements
    - L has ones on diagonal, so det(L) = 1
    - Therefore: det(A) = product of diagonal elements of U

    Note: You can use your lu_decomposition function or implement directly
    """
    raise NotImplementedError
