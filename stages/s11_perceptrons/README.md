# Stage 11: Perceptrons & Activation Functions

## Overview

Build the fundamental building blocks of neural networks: activation functions and the perceptron algorithm. These are the atoms from which all neural networks are made.

**The Big Idea:** A perceptron is the simplest neural network - a single unit that takes weighted inputs, sums them, and passes through an activation function. Modern deep learning is just many perceptrons connected together!

## Learning Philosophy

You will implement **small building blocks** that compose into neural network components:
- 8 activation functions with their derivatives
- The perceptron learning algorithm
- Understanding of linear separability

**Time:** 4-5 hours
**Difficulty:** ⭐⭐

## Getting Started

### Setup

```bash
cd stages/s11_perceptrons

uv run pytest tests/ -v

source .venv/bin/activate
pytest tests/ -v
```

### Files You'll Edit

- `starter/activations.py` - 8 activation functions
- `starter/activation_gradients.py` - Derivatives of activations
- `starter/perceptron.py` - Perceptron algorithm
- Tests in `tests/` verify your implementations

---

## Conceptual Understanding

### Why Activation Functions?

Without activation functions, neural networks would just be linear transformations:

```
Linear network (no activations):
  Layer 1: z₁ = W₁x + b₁
  Layer 2: z₂ = W₂z₁ + b₂
  Combined: z₂ = W₂W₁x + W₂b₁ + b₂ = Wx + b  (still linear!)

With activations:
  Layer 1: a₁ = σ(W₁x + b₁)  ← nonlinearity!
  Layer 2: a₂ = σ(W₂a₁ + b₂)
  Now we can learn nonlinear patterns!
```

### The Zoo of Activation Functions

```
ReLU (Rectified Linear Unit):
  f(x) = max(0, x)

  Most popular! Simple, fast, works well.
  Problem: "Dead neurons" if x always < 0

        |    /
        |   /
   -----+--/---
        |
        |

Leaky ReLU:
  f(x) = x if x > 0 else αx  (α ≈ 0.01)

  Fixes dead neurons by allowing small negative values

       /|   /
      / |  /
   --/--+-/---
        |

Sigmoid:
  f(x) = 1 / (1 + e^(-x))

  Squashes to (0, 1). Used for probabilities.
  Problem: Vanishing gradients at extremes

         ____
        /
   ____/

Tanh:
  f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

  Squashes to (-1, 1). Zero-centered!

       ___
      /
  ___/

GELU (Gaussian Error Linear Unit):
  f(x) = x × Φ(x)  where Φ is CDF of standard normal

  Smooth approximation of ReLU. Used in transformers!
```

### The Perceptron: Simplest Neural Network

```
Inputs      Weights     Sum         Activation    Output
  x₁ ───────── w₁ ─────┐
                        │
  x₂ ───────── w₂ ─────┼──► Σ + b ──► σ(z) ──► ŷ
                        │
  x₃ ───────── w₃ ─────┘

Forward pass:
  z = w₁x₁ + w₂x₂ + w₃x₃ + b
  ŷ = σ(z)

For classification:
  ŷ = 1 if z > 0 else 0
```

### The Perceptron Learning Rule

```
Algorithm:
  1. Initialize weights w = 0, bias b = 0
  2. For each training example (x, y):
     a. Compute prediction: ŷ = sign(w·x + b)
     b. If ŷ ≠ y (mistake):
        - w ← w + η × y × x
        - b ← b + η × y
  3. Repeat until convergence or max iterations

Why it works:
  When ŷ=0 but y=1: w + x makes w·x larger (toward +)
  When ŷ=1 but y=0: w - x makes w·x smaller (toward -)
```

### The XOR Problem: Limits of Perceptrons

```
XOR truth table:
  x₁  x₂  y
   0   0  0
   0   1  1
   1   0  1
   1   1  0

Plot:
        x₂
         │
    0────┼──────1
         │ (1,1)=0
    1    │
   (0,1)=1
         │
   ──────┼────── x₁
         │
    0    │    1
  (0,0)=0    (1,0)=1

No single line can separate 0s from 1s!
This proved that single-layer perceptrons are limited.
Solution: Multiple layers! (s12)
```

## What You'll Build

### Activation Functions (8)

1. `relu(z)` - ReLU: max(0, z)
2. `leaky_relu(z, alpha)` - Leaky ReLU
3. `elu(z, alpha)` - Exponential Linear Unit
4. `gelu(z)` - Gaussian Error Linear Unit
5. `swish(z, beta)` - Swish / SiLU
6. `tanh(z)` - Hyperbolic tangent
7. `softplus(z)` - Smooth ReLU approximation
8. `mish(z)` - Self-regularized activation

### Derivatives (8)

9. `relu_derivative(z)`
10. `leaky_relu_derivative(z, alpha)`
11. `elu_derivative(z, alpha)`
12. `gelu_derivative(z)`
13. `swish_derivative(z, beta)`
14. `tanh_derivative(z)`
15. `softplus_derivative(z)`
16. `mish_derivative(z)`

### Perceptron (4)

17. `perceptron_forward(x, weights, bias)` - Single prediction
18. `perceptron_update(x, y, weights, bias, lr)` - Learning step
19. `perceptron_train(X, y, lr, n_epochs)` - Full training
20. `perceptron_predict(X, weights, bias)` - Batch predictions

## Mathematical Background

### Activation Functions

**ReLU:**
```
f(z) = max(0, z) = { z  if z > 0
                   { 0  if z ≤ 0

f'(z) = { 1  if z > 0
        { 0  if z ≤ 0
```

**Leaky ReLU:**
```
f(z) = { z     if z > 0
       { αz    if z ≤ 0    (typically α = 0.01)

f'(z) = { 1    if z > 0
        { α    if z ≤ 0
```

**ELU (Exponential Linear Unit):**
```
f(z) = { z           if z > 0
       { α(e^z - 1)  if z ≤ 0

f'(z) = { 1          if z > 0
        { f(z) + α   if z ≤ 0
```

**GELU (Gaussian Error Linear Unit):**
```
f(z) = z × Φ(z)  where Φ(z) = 0.5 × (1 + erf(z/√2))

Approximation: f(z) ≈ 0.5z(1 + tanh(√(2/π)(z + 0.044715z³)))

f'(z) = Φ(z) + z × φ(z)  where φ is the standard normal PDF
```

**Swish / SiLU:**
```
f(z) = z × σ(βz)  where σ is sigmoid

f'(z) = σ(βz) + βz × σ(βz) × (1 - σ(βz))
      = σ(βz) + βz × σ(βz) × σ(-βz)
```

**Tanh:**
```
f(z) = (e^z - e^(-z)) / (e^z + e^(-z))

f'(z) = 1 - tanh²(z)
```

**Softplus:**
```
f(z) = log(1 + e^z)

f'(z) = σ(z) = 1 / (1 + e^(-z))

Note: Softplus is a smooth approximation of ReLU!
```

**Mish:**
```
f(z) = z × tanh(softplus(z)) = z × tanh(log(1 + e^z))

f'(z) is complex - see implementation
```

### Perceptron

**Forward pass:**
```
z = Σᵢ wᵢxᵢ + b = w·x + b
ŷ = sign(z) = { 1   if z > 0
              { 0   if z ≤ 0
```

**Update rule:**
```
If y ≠ ŷ:
  w ← w + η × (y - ŷ) × x
  b ← b + η × (y - ŷ)

where η is learning rate, typically 1.0
```

**Convergence theorem:**
If data is linearly separable, perceptron will converge!

## Implementation Guide

### Step 1: Activation Functions (20 min)

```python
def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def tanh(z):
    return np.tanh(z)  # NumPy has it built-in!
```

### Step 2: Derivatives (20 min)

```python
def relu_derivative(z):
    return (z > 0).astype(float)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2
```

### Step 3: Perceptron Algorithm (30 min)

```python
def perceptron_train(X, y, lr=1.0, n_epochs=100):
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0

    for epoch in range(n_epochs):
        for i in range(len(X)):
            z = np.dot(X[i], weights) + bias
            y_pred = 1 if z > 0 else 0

            if y_pred != y[i]:
                weights += lr * (y[i] - y_pred) * X[i]
                bias += lr * (y[i] - y_pred)

    return weights, bias
```

## Key Concepts

### Vanishing Gradients

```
Problem with sigmoid/tanh:
  At extreme values, gradient → 0

  sigmoid'(10) ≈ 0.00005 (basically 0!)
  sigmoid'(-10) ≈ 0.00005

  Gradient updates become tiny in deep networks.

Solution:
  Use ReLU family - gradient is 1 for positive values!
  GELU/Swish - smooth but still well-behaved gradients
```

### Dead Neurons (ReLU)

```
Problem:
  If z < 0 for all inputs, gradient = 0 forever
  Neuron "dies" and never learns

Example:
  Large negative bias makes z always negative

Solutions:
  - Leaky ReLU: small gradient for z < 0
  - ELU: smooth negative region
  - Careful initialization
  - Lower learning rates
```

### When to Use Each Activation

```
Hidden layers:
  - ReLU: Default choice, fast and effective
  - Leaky ReLU: If you have dead neuron problems
  - GELU: Transformers, modern architectures
  - Swish: Good for deeper networks

Output layer:
  - None (linear): Regression
  - Sigmoid: Binary classification
  - Softmax: Multi-class classification
  - Tanh: Output in [-1, 1]
```

## Common Pitfalls

### 1. Numerical Stability in Softplus/Sigmoid

```python
# For large z, exp(z) overflows
def softplus(z):
    return np.log(1 + np.exp(z))  # exp(100) = inf!

# Better: use stable form for large z
def softplus_stable(z):
    return np.where(z > 20, z, np.log(1 + np.exp(z)))
```

### 2. Derivative at Exactly Zero

```python
# ReLU derivative at z=0 is technically undefined
# Convention: use 0 or 1 (doesn't matter much in practice)
def relu_derivative(z):
    return (z > 0).astype(float)  # 0 at z=0
```

### 3. Perceptron on Non-Separable Data

```python
# Perceptron never converges on XOR!
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

weights, bias = perceptron_train(X, y, n_epochs=1000)
# Weights keep oscillating, accuracy never reaches 100%
```

## Experiments to Try

### 1. Visualize Activation Functions

```python
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 100)
activations = {
    'ReLU': relu(z),
    'Leaky ReLU': leaky_relu(z),
    'GELU': gelu(z),
    'Tanh': tanh(z),
}

for name, a in activations.items():
    plt.plot(z, a, label=name)
plt.legend()
plt.grid()
plt.show()
```

### 2. Perceptron Decision Boundary

```python
# After training, plot decision boundary
x1 = np.linspace(0, 1, 100)
# Decision boundary: w1*x1 + w2*x2 + b = 0
# x2 = -(w1*x1 + b) / w2
x2 = -(weights[0] * x1 + bias) / weights[1]
plt.plot(x1, x2, 'r-', label='Decision boundary')
```

### 3. XOR with Hidden Layer

```python
# Preview of s12: XOR becomes separable with hidden layer!
# Hidden layer transforms [0,0],[0,1],[1,0],[1,1]
# into a new space where they ARE separable
```

## Testing Your Implementation

```bash
pytest tests/test_activations.py -v

pytest tests/test_activation_gradients.py -v

pytest tests/test_perceptron.py -v

python scripts/grade.py s11_perceptrons
```

## Real-World Applications

Activation functions are everywhere:

- **GPT/BERT**: GELU activation throughout
- **ResNet/VGG**: ReLU for image classification
- **LSTMs**: Tanh and Sigmoid for gates
- **GANs**: LeakyReLU in discriminator
- **Attention**: Softmax for attention weights

The perceptron algorithm is historically important:
- First trainable neural network (1958)
- Proved limited → "AI Winter"
- Backprop + hidden layers → Deep Learning revolution

## What's Next

After mastering perceptrons:

**s12: Feedforward Networks** - Stack layers with backpropagation
**s13: CNNs** - Specialized architecture for images
**s14: RNNs** - Specialized for sequences

The building blocks you learned here are used in ALL neural networks:
- Every layer applies activations
- Every training uses derivatives
- The perceptron update is gradient descent in disguise!

## Success Criteria

You understand this stage when you can:

- Implement 8 activation functions from formulas
- Derive and implement their derivatives
- Explain vanishing gradients and dead neurons
- Train a perceptron on linearly separable data
- Demonstrate why perceptron fails on XOR
- Choose appropriate activations for different tasks

**Target: All tests passing**

Good luck! These building blocks power all of modern deep learning.
