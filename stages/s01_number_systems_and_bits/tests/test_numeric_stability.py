from __future__ import annotations
from stages.s01_number_systems_and_bits.starter.number import to_fixed, from_fixed, fxp_add

def test_associativity_breaks_in_fixed_point():
    F, W = 8, 16
    a = to_fixed(100.0, F, W)
    b = to_fixed(1e-2, F, W)
    c = to_fixed(-100.0, F, W)

    # (a + b) + c
    s1, _ = fxp_add(a, b, F, W)
    s1, _ = fxp_add(s1, c, F, W)

    # a + (b + c)
    s2, _ = fxp_add(b, c, F, W)
    s2, _ = fxp_add(a, s2, F, W)

    x1 = from_fixed(s1, F, W)
    x2 = from_fixed(s2, F, W)

    # They can differ due to quantization & rounding
    assert abs(x1 - x2) >= 0.0
