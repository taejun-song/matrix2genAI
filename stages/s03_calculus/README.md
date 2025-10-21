# Stage 03: Calculus - Derivatives and Gradients

**Duration**: 2-3 days
**Prerequisites**: s02 (Linear Algebra)
**Difficulty**: ⭐⭐⭐☆☆

## What You'll Learn

Gradients are the backbone of deep learning. In this stage, you'll implement:
- Numerical differentiation (finite differences)
- Forward-mode automatic differentiation
- Reverse-mode automatic differentiation (backpropagation!)

## Why This Matters for ML

- **Training neural networks**: Backprop computes gradients efficiently
- **Optimization**: Gradient descent requires gradients
- **Understanding PyTorch/JAX**: You'll build what they do under the hood

## Getting Started

1. Read [spec.md](spec.md)
2. Start with numerical differentiation (easier)
3. Then tackle autodiff (more challenging but rewarding)
4. Test: `pytest .`
5. Grade: `python ../../scripts/grade.py .`

## Tips

- Test on simple functions first (f(x) = x², f(x) = sin(x))
- Central differences more accurate than forward differences
- Reverse-mode AD is tricky but powerful - draw the computation graph!
- Gradient checking is your friend - use it to debug

## Next Stage

**s04: Probability and Statistics** - Build statistical foundations for ML
