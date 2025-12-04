# Stage 01: Number Systems & Fixed-Point Arithmetic

**Duration**: 1-2 days
**Prerequisites**: Basic Python
**Difficulty**: ⭐⭐☆☆☆

**The Big Idea:** Before we can do machine learning, we need to understand how computers represent numbers. Floating-point isn't magic - it's a clever engineering tradeoff. Fixed-point shows us the fundamentals.

## Getting Started

### Setup

```bash
# Navigate to this stage
cd stages/s01_number_systems_and_bits

# Run tests (using uv - recommended)
uv run pytest tests/ -v

# Or activate venv first
source .venv/bin/activate  # On Unix/macOS
pytest tests/ -v
```

### Files You'll Edit

- `starter/number.py` - Fixed-point arithmetic implementation

### Quick Test Commands

```bash
# Test specific functions
uv run pytest tests/test_number.py -v

# Run all tests
uv run pytest tests/ -v
```

---

## What You'll Learn

Understanding number representation is fundamental to ML. You'll implement:
- Integer bit width, two's complement, overflow/underflow
- Fixed-point encoding/decoding with fractional bits
- Fixed-point add and multiply (wrap vs saturate modes)
- Numerical stability and error analysis

## Conceptual Understanding

### Why Number Representation Matters in ML

**The fundamental problem:** Computers have finite memory.

```
Real numbers:
  • Infinite precision: π = 3.14159265358979323846...
  • Computers can't store infinite digits!

Computer numbers:
  • Finite precision: π ≈ 3.14159274 (8 digits)
  • Approximation introduces errors
  • Errors accumulate in long computations

ML impact:
  • Neural networks: Billions of multiplications
  • Gradient descent: Tiny weight updates
  • Numerical instability can destroy training!
```

### How Do Computers Store Numbers?

**Bits: The fundamental unit**
```
One bit: 0 or 1
  • Can represent 2 states

Two bits: 00, 01, 10, 11
  • Can represent 4 states

n bits: 2^n possible states

Example: 8 bits (1 byte)
  • 2^8 = 256 possible values
  • Unsigned: 0 to 255
  • Signed: -128 to 127
```

### Unsigned Integers: Simple Binary

**Representation:** Each bit position is a power of 2

```
Binary:    1  0  1  1  0  1  0  1
Position:  7  6  5  4  3  2  1  0
Value:    128 0 32 16  0  4  0  1

Decimal value: 128 + 32 + 16 + 4 + 1 = 181

General formula:
  value = Σ(bit_i × 2^i) for i = 0 to n-1

Range for n bits:
  Min: 0 (all bits 0)
  Max: 2^n - 1 (all bits 1)

Example: 8 bits
  00000000 = 0
  11111111 = 255
```

**Problem:** How do we represent negative numbers?

### Two's Complement: Representing Negative Numbers

**Naive approach (doesn't work well):**
```
Sign-magnitude: Use first bit for sign
  0xxx xxxx = positive
  1xxx xxxx = negative

Problems:
  • Two representations of zero: +0 and -0
  • Addition doesn't work: 5 + (-3) requires special logic
```

**Two's complement (the standard):**
```
Idea: Negative numbers "wrap around"

For 8 bits:
  00000000 =   0
  00000001 =   1
  00000010 =   2
  ...
  01111111 = 127  (largest positive)
  10000000 = -128 (most negative)
  10000001 = -127
  ...
  11111110 =  -2
  11111111 =  -1

Pattern: Most significant bit (MSB) = sign bit
  0xxxxxxx = positive (0 to 127)
  1xxxxxxx = negative (-128 to -1)

To negate a number:
  1. Flip all bits (bitwise NOT)
  2. Add 1

Example: Negate 5
  5 = 00000101
  Flip bits: 11111010
  Add 1: 11111011 = -5

Check: What is 11111011 in decimal?
  Method 1: If MSB=1, it's -(flip bits + 1)
    Flip: 00000100
    Add 1: 00000101 = 5
    So 11111011 = -5 ✓

  Method 2: Compute value including sign
    11111011 = -128 + 64 + 32 + 16 + 8 + 2 + 1
             = -128 + 123
             = -5 ✓
```

**Why two's complement is brilliant:**
```
Addition works automatically!

Example: 5 + (-3) = 2
  00000101  (5)
+ 11111101  (-3)
-----------
 100000010  (ignore carry, gives 00000010 = 2) ✓

Example: -5 + (-3) = -8
  11111011  (-5)
+ 11111101  (-3)
-----------
 111111000  (ignore carry, gives 11111000 = -8) ✓

No special logic needed - just binary addition!
```

**Range for n bits (two's complement):**
```
Min: -2^(n-1)
Max:  2^(n-1) - 1

Example: 8 bits
  Min: -2^7 = -128
  Max:  2^7 - 1 = 127

Note the asymmetry: One more negative number than positive!
  This is why abs(-128) overflows in 8 bits
```

### Overflow and Underflow

**Overflow:** Result too large to represent

```
Example: 8-bit signed addition
  127 + 1 = ?

  01111111  (127, max positive)
+ 00000001  (1)
-----------
  10000000  (-128, most negative!)

Wraps around to minimum value!

Real example in ML:
  • Gradient descent step too large
  • Weights explode to infinity
  • Training diverges
```

**Underflow:** Result too small to represent (for fractional numbers)

```
Example: Repeated division
  1.0 / 2 = 0.5
  0.5 / 2 = 0.25
  0.25 / 2 = 0.125
  ...
  Eventually reaches smallest representable value
  Next division → 0 (underflow)

Real example in ML:
  • Softmax with large negative numbers
  • exp(-1000) → 0 (underflow)
  • Division by zero in loss computation
```

### Fixed-Point Arithmetic: A Simpler Alternative to Floating-Point

**Floating-point:** Scientific notation for computers
```
Format: sign × mantissa × 2^exponent

Example (32-bit float):
  1 bit: sign
  8 bits: exponent
  23 bits: mantissa (fractional part)

Good:
  ✓ Huge range: ~10^-38 to 10^38
  ✓ Automatic scaling

Bad:
  ✗ Complex hardware
  ✗ Precision varies with magnitude
  ✗ Rounding errors accumulate
  ✗ Not all numbers representable (0.1 + 0.2 ≠ 0.3!)
```

**Fixed-point:** Dedicate some bits to fractional part
```
Format: [integer bits].[fractional bits]

Example: Q8.8 (8 integer bits, 8 fractional bits)
  Total: 16 bits
  Integer range: -128 to 127
  Fractional precision: 1/256 ≈ 0.0039

Representation:
  Store value × 2^fractional_bits as integer

Example: Store 5.75 in Q8.8
  5.75 × 2^8 = 5.75 × 256 = 1472
  Binary: 00000101.11000000
          ^^^^^^^^ ^^^^^^^^
          int=5    frac=0.75

  Decode: 1472 / 256 = 5.75

How 0.75 works:
  Binary 0.11 = 1×(1/2) + 1×(1/4) = 0.5 + 0.25 = 0.75
```

**Fixed-point arithmetic:**
```
Addition: Just add the integers!
  5.75 + 2.25 = ?

  5.75 → 1472
  2.25 → 576

  1472 + 576 = 2048
  2048 / 256 = 8.0 ✓

Multiplication: Multiply then rescale
  5.75 × 2.0 = ?

  5.75 → 1472
  2.0  → 512

  1472 × 512 = 753,664

  Problem: Result has 2×fractional_bits!
  Need to shift right by fractional_bits

  753,664 >> 8 = 2944
  2944 / 256 = 11.5 ✓

Division: Rescale then divide
  5.75 / 2.0 = ?

  5.75 → 1472
  2.0  → 512

  First scale up dividend:
  (1472 << 8) / 512 = 376,832 / 512 = 736
  736 / 256 = 2.875 ✓
```

**Wrap vs Saturate modes:**
```
Wrap (modulo arithmetic):
  • On overflow, wrap around
  • Example: 127 + 1 = -128
  • Fast (no checks needed)
  • Used in cryptography, hash functions

Saturate (clamp):
  • On overflow, clamp to max/min
  • Example: 127 + 1 = 127
  • Safer for signal processing, ML
  • Used in image processing, audio

ML application:
  • Quantized neural networks use saturating arithmetic
  • Prevents catastrophic failures from overflow
```

### Why Learn Fixed-Point When Everyone Uses Floating-Point?

**Historical reasons:**
```
Before GPUs:
  • Fixed-point was common (DSPs, embedded systems)
  • Simpler hardware, faster computation
  • Trade precision for speed

Modern ML:
  • Quantization: INT8, INT4 for inference
  • Edge devices: Limited power/memory
  • Understanding tradeoffs matters
```

**Pedagogical reasons:**
```
Fixed-point teaches:
  ✓ Bit manipulation
  ✓ Overflow/underflow handling
  ✓ Precision vs range tradeoffs
  ✓ Error accumulation

These concepts apply to floating-point too!
  • Understanding float precision
  • Numerical stability in ML
  • Why batch normalization helps
  • Why gradient clipping is needed
```

**Modern applications:**
```
Quantized neural networks:
  • INT8 inference: 4× faster, 4× less memory
  • Mobile deployment (TensorFlow Lite, PyTorch Mobile)
  • Edge AI (Raspberry Pi, microcontrollers)

Example: ResNet-50
  • Float32: 100 MB, 50 ms/image
  • INT8: 25 MB, 12 ms/image
  • Same accuracy with 4× speedup!
```

### Numerical Stability: Why Small Errors Matter

**Error accumulation:**
```
Example: Sum 1000 numbers

  Perfect precision:
    sum = x₁ + x₂ + ... + x₁₀₀₀

  Fixed-point Q16.16:
    Precision: 1/65536 ≈ 0.000015
    Error per addition: ~0.000015
    After 1000 additions: ~0.015 error

  Order matters!
    sum([1e-10, 1.0, 1e-10]) in float32

    Left to right:
      (1e-10 + 1.0) + 1e-10 = 1.0 + 1e-10 ≈ 1.0 (lost precision!)

    Sorted (small first):
      (1e-10 + 1e-10) + 1.0 = 2e-10 + 1.0 ≈ 1.0 + 2e-10 (better!)

  Kahan summation algorithm: Compensate for lost precision
```

**ML examples:**
```
Vanishing gradients:
  • Deep networks: gradient = ∂L/∂w₁ × ∂L/∂w₂ × ... × ∂L/∂wₙ
  • If each term < 1: product → 0 (underflow!)
  • Solution: Batch normalization, residual connections

Exploding gradients:
  • If each term > 1: product → ∞ (overflow!)
  • Solution: Gradient clipping, careful initialization

Softmax numerical stability:
  • exp(x) for large x → overflow
  • Trick: softmax(x) = softmax(x - max(x))
    exp(x - max) keeps values in reasonable range
```

## Why This Matters

- **Quantization**: Deploy models on mobile/edge devices
- **Numerical stability**: Understand why training fails
- **Hardware**: CPUs/GPUs use these principles
- **Debugging**: Recognize overflow/underflow issues
- **Efficiency**: Lower precision → faster computation

## Tips

- **Start with unsigned integers** - Understand binary first
- **Test edge cases** - Max, min, overflow values
- **Check your math** - Verify fixed-point conversions by hand
- **Use assertions** - Check for overflow in your implementation
- **Read the tests** - They show expected behavior clearly

## Common Pitfalls

1. **Sign extension:** When converting from smaller to larger types
2. **Shift direction:** Left shift (<<) multiplies, right shift (>>) divides
3. **Integer division:** Python 3 uses `/` (float), `//` (integer)
4. **Overflow detection:** Check before operation, not after!

## Success Criteria

You understand this stage when you can:
- ✅ Explain two's complement representation
- ✅ Manually convert numbers to/from fixed-point
- ✅ Predict when overflow will occur
- ✅ Explain wrap vs saturate behavior
- ✅ Implement fixed-point addition and multiplication

**Target: All tests passing**

## Next Stage

Once you achieve 100%, move on to **s02: Linear Algebra** where you'll use these number representations to build matrix operations!
