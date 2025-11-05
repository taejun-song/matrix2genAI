from __future__ import annotations

import numpy as np
import pytest

from stages.s02_linear_algebra.starter.matrix_ops import (
    frobenius_norm,
    matrix_multiply_naive,
    matrix_multiply_vectorized,
    matrix_trace,
    matrix_transpose,
)


class TestMatrixMultiplyNaive:
    def test_basic_multiply(self) -> None:
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = matrix_multiply_naive(A, B)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(result, expected)

    def test_identity_multiply(self) -> None:
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        identity = np.eye(2)
        result = matrix_multiply_naive(A, identity)
        assert np.allclose(result, A)

    def test_non_square_multiply(self) -> None:
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        B = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        result = matrix_multiply_naive(A, B)
        expected = np.dot(A, B)
        assert np.allclose(result, expected)

    def test_dimension_mismatch(self) -> None:
        A = np.array([[1.0, 2.0]])
        B = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError):
            matrix_multiply_naive(A, B)


class TestMatrixMultiplyVectorized:
    def test_basic_multiply(self) -> None:
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = matrix_multiply_vectorized(A, B)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(result, expected)

    def test_matches_naive(self) -> None:
        np.random.seed(42)
        A = np.random.randn(10, 15)
        B = np.random.randn(15, 8)
        naive_result = matrix_multiply_naive(A, B)
        vectorized_result = matrix_multiply_vectorized(A, B)
        assert np.allclose(naive_result, vectorized_result)


class TestMatrixTranspose:
    def test_basic_transpose(self) -> None:
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = matrix_transpose(A)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        assert np.allclose(result, expected)

    def test_square_matrix(self) -> None:
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = matrix_transpose(A)
        expected = np.array([[1.0, 3.0], [2.0, 4.0]])
        assert np.allclose(result, expected)

    def test_double_transpose(self) -> None:
        A = np.random.randn(5, 7)
        result = matrix_transpose(matrix_transpose(A))
        assert np.allclose(result, A)


class TestMatrixTrace:
    def test_basic_trace(self) -> None:
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = matrix_trace(A)
        assert np.isclose(result, 5.0)

    def test_identity_matrix(self) -> None:
        identity = np.eye(5)
        result = matrix_trace(identity)
        assert np.isclose(result, 5.0)

    def test_non_square_raises(self) -> None:
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(ValueError):
            matrix_trace(A)


class TestFrobeniusNorm:
    def test_basic_norm(self) -> None:
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = frobenius_norm(A)
        expected = np.sqrt(1 + 4 + 9 + 16)
        assert np.isclose(result, expected)

    def test_identity_matrix(self) -> None:
        identity = np.eye(3)
        result = frobenius_norm(identity)
        expected = np.sqrt(3.0)
        assert np.isclose(result, expected)

    def test_zero_matrix(self) -> None:
        A = np.zeros((4, 5))
        result = frobenius_norm(A)
        assert np.isclose(result, 0.0)
