# Stage 01 — Number Systems & Fixed-Point Arithmetic

**Goals**
- Understand integer bit width, two's complement, overflow/underflow.
- Implement fixed-point encoding/decoding with a chosen fractional bit count.
- Implement fixed-point add and multiply, with configurable wrap vs. saturate behavior.
- Demonstrate numerical stability and error bounds vs. float.

**Done =**
- All tests in `tests/` pass: `uv run pytest stages/01_number_systems_and_bits -q`.
- A short note in `starter/train_demo.py` prints a summary and doesn't crash.

**Hand-in**
- You only edit `starter/number.py` (marked with TODOs).

