from __future__ import annotations

import numpy as np

from stages.s05_optimization.starter.schedules import (
    cosine_annealing,
    exponential_decay,
    step_decay,
)


def test_step_decay() -> None:
    lr0 = 0.1
    lr10 = step_decay(lr0, 10, drop_rate=0.5, epochs_drop=10)
    assert np.isclose(lr10, 0.05)


def test_exponential_decay() -> None:
    lr0 = 0.1
    lr10 = exponential_decay(lr0, 10, decay_rate=0.9)
    expected = 0.1 * (0.9 ** 10)
    assert np.isclose(lr10, expected)


def test_cosine_annealing() -> None:
    lr0 = 0.1
    lr50 = cosine_annealing(lr0, 50, T_max=100)
    assert 0 < lr50 < lr0
