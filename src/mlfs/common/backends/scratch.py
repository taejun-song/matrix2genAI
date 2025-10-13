from __future__ import annotations

import numpy as np

from mlfs.common.tensor import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(np.add(a.data, b.data))


def mul(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(np.multiply(a.data, b.data))
