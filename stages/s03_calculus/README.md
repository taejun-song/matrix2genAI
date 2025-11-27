# Stage 03: Calculus - Derivatives and Gradients

**Duration**: 2-3 days
**Prerequisites**: s02 (Linear Algebra)
**Difficulty**: ⭐⭐⭐☆☆

**The Big Idea:** Derivatives tell us how to improve. Gradients point toward better model parameters. Without them, there's no learning!

## Getting Started

### Setup

```bash
# Navigate to this stage
cd stages/s03_calculus

# Run tests (using uv - recommended)
uv run pytest tests/ -v

# Or activate venv first
source .venv/bin/activate  # On Unix/macOS
pytest tests/ -v
```

### Files You'll Edit

- `starter/numerical_diff.py` - Finite differences, gradient checking
- `starter/forward_ad.py` - Forward-mode automatic differentiation
- `starter/reverse_ad.py` - Reverse-mode AD (backpropagation)

### Quick Test Commands

```bash
# Test specific module
uv run pytest tests/test_numerical_diff.py -v

# Test specific function
uv run pytest tests/test_reverse_ad.py::TestBackprop -v

# Run all tests
uv run pytest tests/ -v
```

---

## What You'll Learn

Gradients are the backbone of deep learning. In this stage, you'll implement:
- Numerical differentiation (finite differences)
- Forward-mode automatic differentiation
- Reverse-mode automatic differentiation (backpropagation!)

## Conceptual Understanding

### What is a Derivative?

**Intuition:** The derivative tells you how much the output changes when you nudge the input.

```
Example: Position and speed
  Position at time t: s(t) = t²

  At t=3: s(3) = 9 meters
  At t=3.01: s(3.01) = 9.0601 meters

  Change in position: 9.0601 - 9 = 0.0601 meters
  Change in time: 0.01 seconds
  Speed ≈ 0.0601 / 0.01 = 6.01 m/s

  As we make the nudge smaller, we get closer to the true derivative:
  s'(3) = 2·3 = 6 m/s
```

**ML application:**
```
Loss function: L(w) = (prediction - actual)²

Current weight: w = 1.5
L(1.5) = 4.0

Question: Should we increase or decrease w to reduce loss?

Answer: Compute derivative dL/dw
  • If dL/dw > 0: Loss increases with w → decrease w
  • If dL/dw < 0: Loss decreases with w → increase w
  • If dL/dw = 0: We're at a minimum!

This is gradient descent!
```

### Why Not Just Compute Derivatives by Hand?

**Problem:** Neural networks have millions of parameters!

```
Simple network:
  Input layer: 784 neurons (28×28 image)
  Hidden layer: 128 neurons
  Output layer: 10 neurons (digits 0-9)

  Number of weights: 784×128 + 128×10 = 101,632

Computing derivatives by hand: IMPOSSIBLE
Writing derivative formulas for each layer: TEDIOUS and ERROR-PRONE

Solution: Automatic differentiation!
```

### Three Ways to Compute Derivatives

**1. Symbolic Differentiation (high school calculus)**
```
f(x) = x³ + 2x

By hand: f'(x) = 3x² + 2

✓ Exact
✗ Requires knowing calculus rules
✗ Expressions can explode in size
✗ Not practical for complex ML models
```

**2. Numerical Differentiation (approximate)**
```
f'(x) ≈ [f(x + h) - f(x)] / h

Example: f(x) = x³ at x=2
  f(2.001) = 8.012006001
  f(2) = 8
  f'(2) ≈ (8.012006001 - 8) / 0.001 = 12.006

True value: f'(2) = 3·2² = 12

✓ Easy to implement
✓ Works for any function
✗ Approximate (has error)
✗ Slow (requires multiple function evaluations)
✗ Numerically unstable for very small h
```

**3. Automatic Differentiation (exact and efficient!)**
```
Forward mode: Compute derivatives alongside function values
Reverse mode: Compute all derivatives in one backward pass

✓ Exact (to machine precision)
✓ Efficient
✓ Works for any code (loops, conditionals, etc.)
✓ This is what PyTorch and JAX use!
```

### The Chain Rule: Foundation of Backpropagation

**One-variable chain rule:**
```
If y = f(g(x)), then dy/dx = (df/dg) · (dg/dx)

Example: y = sin(x²)
  Let u = x², then y = sin(u)

  dy/du = cos(u)
  du/dx = 2x

  dy/dx = cos(u) · 2x = cos(x²) · 2x
```

**Why this matters for neural networks:**
```
Neural network as composition of functions:

  Input x → Layer 1 → Layer 2 → Layer 3 → Loss L

  L = f₃(f₂(f₁(x)))

To compute dL/dx, we use chain rule:
  dL/dx = (dL/df₃) · (df₃/df₂) · (df₂/df₁) · (df₁/dx)

This is backpropagation!
```

### Forward vs Reverse Mode: A Crucial Distinction

**Forward Mode:**
```
Compute derivatives from inputs to outputs

Example: f(x₁, x₂) = x₁·x₂ + sin(x₁)

To get ∂f/∂x₁ and ∂f/∂x₂, need TWO forward passes:
  1. Forward pass with "seed" ∂x₁ = 1, ∂x₂ = 0 → gives ∂f/∂x₁
  2. Forward pass with "seed" ∂x₁ = 0, ∂x₂ = 1 → gives ∂f/∂x₂

Cost: O(n_inputs) passes for all gradients

Good for: Few inputs, many outputs (e.g., evaluating Jacobian-vector product)
```

**Reverse Mode (Backpropagation):**
```
Compute derivatives from outputs to inputs

Same example: f(x₁, x₂) = x₁·x₂ + sin(x₁)

ONE backward pass gives both ∂f/∂x₁ AND ∂f/∂x₂!

Cost: O(n_outputs) passes for all gradients

Good for: Many inputs, few outputs (e.g., neural network loss)

This is why deep learning uses reverse mode!
```

**The key insight:**
```
Neural network:
  Inputs: 1,000,000 parameters
  Output: 1 scalar loss value

Forward mode: 1,000,000 passes needed ❌
Reverse mode: 1 backward pass needed ✓

Reverse mode is 1,000,000× faster!
```

### Computational Graphs: Visualizing Computation

**Example:** f(x, y) = x·y + sin(x)

```
Computation graph:

  x ──┬──→ multiply ──→ add ──→ f
      │      ↑           ↑
      │      │           │
  y ──┘      └─── sin ───┘

Forward pass (compute values):
  v₁ = x
  v₂ = y
  v₃ = v₁ · v₂
  v₄ = sin(v₁)
  v₅ = v₃ + v₄

Backward pass (compute gradients):
  ∂f/∂v₅ = 1           (output gradient)
  ∂f/∂v₄ = 1           (from +)
  ∂f/∂v₃ = 1           (from +)
  ∂f/∂v₁ = v₂ + cos(v₁) (from · and sin)
  ∂f/∂v₂ = v₁          (from ·)
```

**The magic:** Each operation knows how to compute its local gradient!

## Why This Matters for ML

- **Training neural networks**: Backprop computes gradients efficiently
- **Optimization**: Gradient descent requires gradients
- **Understanding PyTorch/JAX**: You'll build what they do under the hood

## Tips

- Test on simple functions first (f(x) = x², f(x) = sin(x))
- Central differences more accurate than forward differences
- Reverse-mode AD is tricky but powerful - draw the computation graph!
- Gradient checking is your friend - use it to debug

## Next Stage

**s04: Probability and Statistics** - Build statistical foundations for ML
