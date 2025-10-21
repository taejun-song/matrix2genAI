# Stage 03: Calculus - Derivatives and Gradients

## Overview

Implement numerical differentiation and automatic differentiation (autodiff). These are the foundation of backpropagation in neural networks.

## Mathematical Background

### Derivatives
- Derivative: **f'(x) = lim[h→0] (f(x+h) - f(x)) / h**
- Measures rate of change of function
- Used for optimization (finding minima/maxima)

### Gradients
- Gradient: **∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]**
- Vector of partial derivatives
- Points in direction of steepest ascent

### Chain Rule
- **(f ∘ g)'(x) = f'(g(x)) × g'(x)**
- Fundamental for backpropagation
- Allows computing derivatives of composed functions

## Tasks

### Part 1: Numerical Differentiation

Implement in `starter/numerical_diff.py`:

1. **`finite_difference(f, x, h=1e-5, method='central')`**
   - Compute derivative using finite differences
   - Forward: (f(x+h) - f(x)) / h
   - Central: (f(x+h) - f(x-h)) / (2h) (more accurate)
   - Backward: (f(x) - f(x-h)) / h

2. **`gradient_finite_diff(f, x, h=1e-5)`**
   - Compute gradient ∇f at point x using finite differences
   - x is a vector, return vector of partial derivatives

3. **`jacobian_finite_diff(f, x, h=1e-5)`**
   - Compute Jacobian matrix for vector-valued function f: ℝⁿ → ℝᵐ
   - J[i,j] = ∂f_i/∂x_j

4. **`hessian_finite_diff(f, x, h=1e-5)`**
   - Compute Hessian matrix (matrix of second derivatives)
   - H[i,j] = ∂²f/∂x_i∂x_j

5. **`gradient_check(f, grad_f, x, epsilon=1e-5, tolerance=1e-7)`**
   - Check analytical gradient against numerical gradient
   - Return True if they match within tolerance

### Part 2: Automatic Differentiation (Forward Mode)

Implement in `starter/autodiff_forward.py`:

1. **`class Dual`** - Dual number for forward-mode AD
   - Stores value and derivative
   - Overload operators: +, -, *, /, **
   - Implement: sin, cos, exp, log

2. **`forward_mode_gradient(f, x)`**
   - Compute gradient using forward-mode AD
   - Run f multiple times (once per input dimension)

### Part 3: Automatic Differentiation (Reverse Mode)

Implement in `starter/autodiff_reverse.py`:

1. **`class Variable`** - Computational graph node
   - Stores value, gradient, and backward function
   - Build computation graph during forward pass

2. **`class Operations`** - Define operations with backward pass
   - Add, Multiply, Sin, Exp, etc.
   - Each operation knows how to backpropagate gradients

3. **`reverse_mode_gradient(f, x)`**
   - Compute gradient using reverse-mode AD (backpropagation)
   - More efficient than forward mode for functions ℝⁿ → ℝ

## Constraints

- For numerical differentiation: use NumPy
- For autodiff: implement from scratch (no PyTorch/JAX)
- Handle scalar and vector inputs
- Maintain numerical stability (choice of h)

## Numerical Considerations

1. **Step size h**: Too large → truncation error, too small → rounding error
2. **Central differences**: More accurate than forward/backward (O(h²) vs O(h))
3. **Condition number**: Poorly conditioned functions harder to differentiate

## Expected Performance

- Numerical differentiation: O(n) function evaluations for gradient
- Forward-mode AD: O(n) passes through function
- Reverse-mode AD: O(1) backward pass (constant overhead)

## Testing

Tests verify:
- Correctness on polynomials, trig functions, compositions
- Gradient checking matches analytical solutions
- Autodiff matches numerical differentiation
- Chain rule works correctly
- Edge cases (zero gradients, nested functions)

## Learning Goals

- Understand how gradients are computed numerically
- Learn the difference between forward and reverse-mode AD
- Build foundation for implementing backpropagation
- Appreciate why reverse-mode AD (backprop) is efficient for deep learning
