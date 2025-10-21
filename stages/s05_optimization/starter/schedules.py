from __future__ import annotations

import numpy as np


def step_decay(lr0: float, epoch: int, drop_rate: float = 0.5, epochs_drop: int = 10) -> float:
    """TODO: Step decay learning rate schedule."""
    raise NotImplementedError


def exponential_decay(lr0: float, epoch: int, decay_rate: float = 0.95) -> float:
    """TODO: Exponential decay learning rate schedule."""
    raise NotImplementedError


def cosine_annealing(lr0: float, epoch: int, T_max: int = 100) -> float:
    """TODO: Cosine annealing learning rate schedule."""
    raise NotImplementedError
