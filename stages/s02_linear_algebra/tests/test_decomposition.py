from __future__ import annotations

import numpy as np
import pytest

from stages.s02_linear_algebra.starter.decomposition import (
    determinant_lu,
    determinant_recursive,
    power_iteration,
    qr_decomposition_gram_schmidt,
)


class TestQRDecomposition:
    def test_basic_qr(self) -> None:
        A = np.array([[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]])
        Q, R = qr_decomposition_gram_schmidt(A)
        assert np.allclose(Q @ R, A, atol=1e-10)
        assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-10)
        for i in range(3):
            for j in range(i):
                assert np.isclose(R[i, j], 0.0, atol=1e-10)

    def test_orthonormality(self) -> None:
        np.random.seed(42)
        A = np.random.randn(5, 3)
        Q, R = qr_decomposition_gram_schmidt(A)
        assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-10)

    def test_tall_matrix(self) -> None:
        A = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        Q, R = qr_decomposition_gram_schmidt(A)
        assert np.allclose(Q @ R, A, atol=1e-10)


class TestPowerIteration:
    def test_symmetric_matrix(self) -> None:
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        eigenvalue, eigenvector = power_iteration(A)
        assert np.allclose(A @ eigenvector, eigenvalue * eigenvector, atol=1e-6)
        assert np.isclose(np.linalg.norm(eigenvector), 1.0)

    def test_3x3_matrix(self) -> None:
        A = np.array([[6.0, -2.0, 2.0], [-2.0, 3.0, -1.0], [2.0, -1.0, 3.0]])
        eigenvalue, eigenvector = power_iteration(A, num_iters=200)
        assert np.allclose(A @ eigenvector, eigenvalue * eigenvector, atol=1e-5)

    def test_dominant_eigenvalue_found(self) -> None:
        np.random.seed(42)
        true_eigenvalues = np.array([10.0, 3.0, 1.0])
        V, _ = np.linalg.qr(np.random.randn(3, 3))
        A = V @ np.diag(true_eigenvalues) @ V.T
        eigenvalue, eigenvector = power_iteration(A, num_iters=200)
        assert np.isclose(abs(eigenvalue), 10.0, atol=1e-4)


class TestDeterminantRecursive:
    def test_2x2_matrix(self) -> None:
        A = np.array([[3.0, 8.0], [4.0, 6.0]])
        result = determinant_recursive(A)
        expected = 3*6 - 8*4
        assert np.isclose(result, expected)

    def test_3x3_matrix(self) -> None:
        A = np.array([[6.0, 1.0, 1.0], [4.0, -2.0, 5.0], [2.0, 8.0, 7.0]])
        result = determinant_recursive(A)
        expected = np.linalg.det(A)
        assert np.isclose(result, expected, atol=1e-10)

    def test_identity_matrix(self) -> None:
        A = np.eye(3)
        result = determinant_recursive(A)
        assert np.isclose(result, 1.0)

    def test_singular_matrix(self) -> None:
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        result = determinant_recursive(A)
        assert np.isclose(result, 0.0, atol=1e-10)


class TestDeterminantLU:
    def test_2x2_matrix(self) -> None:
        A = np.array([[3.0, 8.0], [4.0, 6.0]])
        result = determinant_lu(A)
        expected = 3*6 - 8*4
        assert np.isclose(result, expected, atol=1e-10)

    def test_3x3_matrix(self) -> None:
        A = np.array([[6.0, 1.0, 1.0], [4.0, -2.0, 5.0], [2.0, 8.0, 7.0]])
        result = determinant_lu(A)
        expected = np.linalg.det(A)
        assert np.isclose(result, expected, atol=1e-9)

    def test_larger_matrix(self) -> None:
        np.random.seed(42)
        A = np.random.randn(6, 6)
        result = determinant_lu(A)
        expected = np.linalg.det(A)
        assert np.isclose(result, expected, atol=1e-8)

    def test_matches_recursive(self) -> None:
        A = np.array([[2.0, 3.0, 1.0], [1.0, 4.0, 2.0], [3.0, 1.0, 5.0]])
        det_recursive = determinant_recursive(A)
        det_lu = determinant_lu(A)
        assert np.isclose(det_recursive, det_lu, atol=1e-10)
