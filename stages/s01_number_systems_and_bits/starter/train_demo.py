from __future__ import annotations
from .number import to_fixed, from_fixed, fxp_add, fxp_mul

def main() -> None:
    F, W = 8, 16
    a = to_fixed(1.25, F, W)
    b = to_fixed(-0.75, F, W)
    s, ov1 = fxp_add(a, b, F, W, mode="wrap")
    p, ov2 = fxp_mul(a, b, F, W, mode="wrap")
    print("a:", a, "->", from_fixed(a, F, W))
    print("b:", b, "->", from_fixed(b, F, W))
    print("sum:", s, "ovf?", ov1, "->", from_fixed(s, F, W))
    print("prod:", p, "ovf?", ov2, "->", from_fixed(p, F, W))

if __name__ == "__main__":
    main()

