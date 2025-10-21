from __future__ import annotations

import numpy as np

from stages.s04_probability.starter.estimation import mle_bernoulli, mle_normal


def test_mle_normal() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean_est, std_est = mle_normal(x)
    assert np.isclose(mean_est, 3.0)
    assert np.isclose(std_est, np.std(x))


def test_mle_bernoulli() -> None:
    x = np.array([1, 1, 0, 1, 0, 1, 1])
    p_est = mle_bernoulli(x)
    assert np.isclose(p_est, 5/7)
