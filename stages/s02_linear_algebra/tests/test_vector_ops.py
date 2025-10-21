from __future__ import annotations

import numpy as np
import pytest

from stages.s02_linear_algebra.starter.vector_ops import (
    cosine_similarity,
    dot_product,
    vector_add,
    vector_norm,
    vector_scalar_multiply,
)


class TestVectorAdd:
    def test_basic_addition(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        result = vector_add(v, w)
        expected = np.array([5.0, 7.0, 9.0])
        assert np.allclose(result, expected)

    def test_dimension_mismatch(self) -> None:
        v = np.array([1.0, 2.0])
        w = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            vector_add(v, w)

    def test_zero_vector(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        w = np.zeros(3)
        result = vector_add(v, w)
        assert np.allclose(result, v)


class TestVectorScalarMultiply:
    def test_basic_multiply(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        result = vector_scalar_multiply(v, 2.0)
        expected = np.array([2.0, 4.0, 6.0])
        assert np.allclose(result, expected)

    def test_zero_scalar(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        result = vector_scalar_multiply(v, 0.0)
        assert np.allclose(result, np.zeros(3))

    def test_negative_scalar(self) -> None:
        v = np.array([1.0, -2.0, 3.0])
        result = vector_scalar_multiply(v, -1.0)
        expected = np.array([-1.0, 2.0, -3.0])
        assert np.allclose(result, expected)


class TestDotProduct:
    def test_basic_dot_product(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        result = dot_product(v, w)
        expected = 1*4 + 2*5 + 3*6
        assert np.isclose(result, expected)

    def test_orthogonal_vectors(self) -> None:
        v = np.array([1.0, 0.0])
        w = np.array([0.0, 1.0])
        result = dot_product(v, w)
        assert np.isclose(result, 0.0)

    def test_dimension_mismatch(self) -> None:
        v = np.array([1.0, 2.0])
        w = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            dot_product(v, w)


class TestVectorNorm:
    def test_l2_norm(self) -> None:
        v = np.array([3.0, 4.0])
        result = vector_norm(v, p=2.0)
        assert np.isclose(result, 5.0)

    def test_l1_norm(self) -> None:
        v = np.array([3.0, -4.0])
        result = vector_norm(v, p=1.0)
        assert np.isclose(result, 7.0)

    def test_inf_norm(self) -> None:
        v = np.array([3.0, -7.0, 2.0])
        result = vector_norm(v, p=np.inf)
        assert np.isclose(result, 7.0)

    def test_zero_vector(self) -> None:
        v = np.zeros(5)
        result = vector_norm(v)
        assert np.isclose(result, 0.0)


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        result = cosine_similarity(v, v)
        assert np.isclose(result, 1.0)

    def test_opposite_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        w = -v
        result = cosine_similarity(v, w)
        assert np.isclose(result, -1.0)

    def test_orthogonal_vectors(self) -> None:
        v = np.array([1.0, 0.0, 0.0])
        w = np.array([0.0, 1.0, 0.0])
        result = cosine_similarity(v, w)
        assert np.isclose(result, 0.0)

    def test_zero_vector(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        w = np.zeros(3)
        result = cosine_similarity(v, w)
        assert np.isclose(result, 0.0)

    def test_dimension_mismatch(self) -> None:
        v = np.array([1.0, 2.0])
        w = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            cosine_similarity(v, w)
