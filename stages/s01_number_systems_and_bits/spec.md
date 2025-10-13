## Definitions

- We encode real `x` to fixed point integer `n = round(x * 2^F)`, with `F` fractional bits.
- Decode `x̂ = n / 2^F`.
- Use signed two's complement arithmetic with total bit width `W` (default 16).

## Tasks

1. Implement:
   - `to_fixed(x, F, W)`
   - `from_fixed(n, F, W)`
   - `fxp_add(a, b, F, W, mode="wrap"|"saturate")`
   - `fxp_mul(a, b, F, W, mode="wrap"|"saturate")`
   - Overflow detection flags

2. Prove bounds on quantization error:
   - `|x - round(x * 2^F) / 2^F| ≤ 2^-(F+1)`

3. Show via tests that error decreases as `F` increases.

