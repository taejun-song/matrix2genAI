from __future__ import annotations

import numpy as np

from stages.s04_probability.starter.statistics import covariance, covariance_matrix, mean, variance


def test_mean() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(mean(x), 3.0)


def test_variance() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = variance(x, ddof=1)
    expected = np.var(x, ddof=1)
    assert np.isclose(result, expected)


def test_covariance() -> None:
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    result = covariance(x, y)
    expected = np.cov(x, y)[0, 1]
    assert np.isclose(result, expected)


def test_covariance_matrix() -> None:
    X = np.random.randn(100, 3)
    result = covariance_matrix(X)
    expected = np.cov(X.T)
    assert np.allclose(result, expected, atol=1e-10)
