from __future__ import annotations

import numpy as np
import pytest

from stages.s02_linear_algebra.starter.linear_systems import (
    backward_substitution,
    forward_substitution,
    gaussian_elimination,
    lu_decomposition,
)


class TestForwardSubstitution:
    def test_basic_solve(self) -> None:
        L = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [4.0, 1.0, 2.0]])
        b = np.array([2.0, 5.0, 6.0])
        x = forward_substitution(L, b)
        assert np.allclose(L @ x, b)

    def test_identity_matrix(self) -> None:
        L = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        x = forward_substitution(L, b)
        assert np.allclose(x, b)

    def test_singular_raises(self) -> None:
        L = np.array([[1.0, 0.0], [2.0, 0.0]])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            forward_substitution(L, b)


class TestBackwardSubstitution:
    def test_basic_solve(self) -> None:
        U = np.array([[2.0, 1.0, 4.0], [0.0, 3.0, 1.0], [0.0, 0.0, 2.0]])
        b = np.array([10.0, 7.0, 4.0])
        x = backward_substitution(U, b)
        assert np.allclose(U @ x, b)

    def test_identity_matrix(self) -> None:
        U = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        x = backward_substitution(U, b)
        assert np.allclose(x, b)


class TestGaussianElimination:
    def test_basic_solve(self) -> None:
        A = np.array([[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]])
        b = np.array([1.0, -2.0, 0.0])
        x = gaussian_elimination(A, b)
        assert np.allclose(A @ x, b, atol=1e-10)

    def test_identity_system(self) -> None:
        A = np.eye(4)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = gaussian_elimination(A, b)
        assert np.allclose(x, b)

    def test_larger_system(self) -> None:
        np.random.seed(42)
        n = 10
        A = np.random.randn(n, n)
        x_true = np.random.randn(n)
        b = A @ x_true
        x = gaussian_elimination(A, b)
        assert np.allclose(x, x_true, atol=1e-8)

    def test_singular_raises(self) -> None:
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            gaussian_elimination(A, b)


class TestLUDecomposition:
    def test_basic_decomposition(self) -> None:
        A = np.array([[4.0, 3.0], [6.0, 3.0]])
        L, U = lu_decomposition(A)
        assert np.allclose(L @ U, A)
        assert np.allclose(np.diag(L), np.ones(2))

    def test_3x3_decomposition(self) -> None:
        A = np.array([[2.0, -1.0, -2.0], [-4.0, 6.0, 3.0], [-4.0, -2.0, 8.0]])
        L, U = lu_decomposition(A)
        assert np.allclose(L @ U, A)
        for i in range(3):
            assert np.isclose(L[i, i], 1.0)
            for j in range(i):
                assert np.isclose(U[i, j], 0.0)

    def test_identity_matrix(self) -> None:
        A = np.eye(3)
        L, U = lu_decomposition(A)
        assert np.allclose(L, np.eye(3))
        assert np.allclose(U, np.eye(3))

    def test_random_matrix(self) -> None:
        np.random.seed(42)
        A = np.random.randn(5, 5)
        L, U = lu_decomposition(A)
        assert np.allclose(L @ U, A, atol=1e-10)
