from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ArrayLike = Any


@dataclass
class Tensor:
    """Minimal Tensor wrapper for early stages (no autograd yet)."""

    data: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def astype(self, dtype: np.dtype) -> Tensor:
        return Tensor(self.data.astype(dtype))

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, dtype={self.data.dtype})"
