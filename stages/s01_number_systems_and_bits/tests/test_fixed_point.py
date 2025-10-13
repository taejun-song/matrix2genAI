from __future__ import annotations
import math
import random
import pytest
from stages.01_number_systems_and_bits.starter.number import (
    to_fixed, from_fixed, fxp_add, fxp_mul
)

def test_roundtrip_small_values():
    F, W = 8, 16
    for x in [0.0, 0.5, -0.5, 1.25, -1.75, 3.125]:
        n = to_fixed(x, F, W)
        xr = from_fixed(n, F, W)
        assert abs(x - xr) <= 2 ** (-(F + 1)) + 1e-12

def test_add_basic():
    F, W = 8, 16
    a = to_fixed(0.75, F, W)
    b = to_fixed(0.50, F, W)
    s, ov = fxp_add(a, b, F, W)
    assert not ov
    assert abs(from_fixed(s, F, W) - 1.25) <= 2 ** (-(F + 1)) + 1e-12

def test_mul_basic():
    F, W = 8, 16
    a = to_fixed(1.5, F, W)
    b = to_fixed(-0.5, F, W)
    p, ov = fxp_mul(a, b, F, W)
    assert not ov
    assert abs(from_fixed(p, F, W) - (-0.75)) <= 2 ** (-(F + 1)) + 1e-12

def test_saturation_overflow():
    F, W = 8, 16
    big = to_fixed(100.0, F, W)
    s, ov = fxp_add(big, big, F, W, mode="saturate")
    assert ov
    # should be clamped to max representable
    hi = (2 ** (W - 1) - 1) / (2 ** F)
    assert abs(from_fixed(s, F, W) - hi) <= 1e-6

@pytest.mark.parametrize("F", [4, 8, 12])
def test_error_shrinks_with_F(F: int):
    W = 16
    random.seed(0)
    errs = []
    for _ in range(200):
        x = random.uniform(-1.5, 1.5)
        n = to_fixed(x, F, W)
        xr = from_fixed(n, F, W)
        errs.append(abs(x - xr))
    assert sum(errs) / len(errs) < 0.5 * (2 ** (-(F - 1)))

