from __future__ import annotations
from typing import Tuple, Literal

Mode = Literal["wrap", "saturate"]


def _mask(width: int) -> int:
    return (1 << width) - 1


def _twos_to_signed(n: int, width: int) -> int:
    """Interpret n as signed two's complement with given width."""
    sign_bit = 1 << (width - 1)
    return (n & (sign_bit - 1)) - (n & sign_bit)


def _signed_to_twos(n: int, width: int) -> int:
    """Map Python int n (assumed in range) to two's complement within width."""
    return n & _mask(width)


def to_fixed(x: float, F: int, W: int = 16) -> int:
    """TODO: encode real x to fixed-point integer with F fractional bits in W total bits."""
    # 1) scale
    # 2) round to nearest int
    # 3) clamp to representable range for signed W-bit two's complement
    # 4) return as two's complement (masked)
    raise NotImplementedError


def from_fixed(n: int, F: int, W: int = 16) -> float:
    """TODO: decode fixed-point integer n (two's complement, W bits) back to float."""
    # 1) interpret n as signed W-bit
    # 2) scale back by 2^F
    raise NotImplementedError


def _saturate_signed(n: int, W: int) -> int:
    lo = -(1 << (W - 1))
    hi = (1 << (W - 1)) - 1
    return max(lo, min(hi, n))


def fxp_add(a: int, b: int, F: int, W: int = 16, mode: Mode = "wrap") -> Tuple[int, bool]:
    """TODO: fixed-point add. Return (sum, overflow_flag). a,b are W-bit two's complement ints."""
    # 1) interpret a,b as signed
    # 2) compute s = a + b
    # 3) detect overflow (signs of inputs and result)
    # 4) wrap or saturate to W-bit
    raise NotImplementedError


def fxp_mul(a: int, b: int, F: int, W: int = 16, mode: Mode = "wrap") -> Tuple[int, bool]:
    """
    TODO: fixed-point multiply.
    a,b are W-bit two's complement ints representing values with F fractional bits.
    Return (prod, overflow_flag) in the same fixed-point format.
    """
    # Hint: (a_real * b_real) = (a * b) / 2^F in integer domain.
    # Compute wide product, then shift, then wrap/saturate.
    raise NotImplementedError

