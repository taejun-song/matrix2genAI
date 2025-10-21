from __future__ import annotations

import numpy as np
from scipy import stats

from stages.s04_probability.starter.distributions import (
    bernoulli_pmf,
    bernoulli_sample,
    normal_pdf,
    normal_sample,
)


def test_normal_pdf() -> None:
    x = np.array([0.0, 1.0, -1.0])
    result = normal_pdf(x, mean=0.0, std=1.0)
    expected = stats.norm.pdf(x)
    assert np.allclose(result, expected)


def test_normal_sample() -> None:
    samples = normal_sample(mean=0.0, std=1.0, size=10000)
    assert np.abs(np.mean(samples)) < 0.1
    assert np.abs(np.std(samples) - 1.0) < 0.1


def test_bernoulli_pmf() -> None:
    k = np.array([0, 1])
    result = bernoulli_pmf(k, p=0.7)
    expected = stats.bernoulli.pmf(k, p=0.7)
    assert np.allclose(result, expected)


def test_bernoulli_sample() -> None:
    samples = bernoulli_sample(p=0.7, size=10000)
    assert np.abs(np.mean(samples) - 0.7) < 0.05
