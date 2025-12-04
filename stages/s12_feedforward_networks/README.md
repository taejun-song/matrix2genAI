# Stage 12: Feedforward Networks & Backpropagation

## Overview

Build a complete neural network framework from scratch. Learn how layers compose, how gradients flow backward through the network, and how to train deep models.

**The Big Idea:** Neural networks are function compositions. Backpropagation uses the chain rule to compute gradients efficiently, layer by layer, from output to input.

## Learning Philosophy

You will implement:
- Weight initialization (Xavier, He)
- Dense layers with forward and backward passes
- Neural network class for arbitrary architectures
- Training utilities (batching, early stopping)

**Time:** 8-10 hours
**Difficulty:** ⭐⭐⭐⭐

## Getting Started

```bash
cd stages/s12_feedforward_networks
uv run pytest tests/ -v
```

### Files You'll Edit

- `starter/weight_init.py` - Initialization strategies
- `starter/layer.py` - Dense layer implementation
- `starter/neural_network.py` - Network class
- `starter/training.py` - Training utilities

---

## Conceptual Understanding

### Forward Pass

```
Input → Layer 1 → Activation → Layer 2 → ... → Output

x ─→ [W₁x + b₁] ─→ [ReLU] ─→ [W₂a₁ + b₂] ─→ [Softmax] ─→ ŷ
        z₁           a₁           z₂              ŷ

Each layer: z = Wx + b, then a = activation(z)
```

### Backward Pass (Backpropagation)

```
The Chain Rule in Action:

∂L/∂W₁ = ∂L/∂ŷ × ∂ŷ/∂z₂ × ∂z₂/∂a₁ × ∂a₁/∂z₁ × ∂z₁/∂W₁

Compute gradients from output → input:
1. ∂L/∂ŷ: Loss gradient (how loss changes with prediction)
2. ∂ŷ/∂z: Activation gradient (softmax derivative)
3. ∂z/∂W: Linear gradient (equals input!)
4. Repeat for each layer backward
```

### Weight Initialization

**Why it matters:**
```
Too small weights → Gradients vanish
Too large weights → Gradients explode

Xavier (Glorot):
  W ~ U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
  Good for tanh, sigmoid

He:
  W ~ N(0, √(2/fan_in))
  Good for ReLU
```

## What You'll Build

### Weight Initialization (5 functions)
1. `xavier_uniform(fan_in, fan_out)` - Glorot uniform
2. `xavier_normal(fan_in, fan_out)` - Glorot normal
3. `he_uniform(fan_in)` - He uniform
4. `he_normal(fan_in)` - He normal
5. `zeros(shape)` - Zero init for biases

### Layer (1 class)
6. `DenseLayer` - forward, backward, get_params, set_params

### Neural Network (4 functions)
7. `NeuralNetwork` - add_layer, forward, backward, train_step
8. `compute_loss(y_true, y_pred, loss_type)`
9. `compute_loss_gradient(y_true, y_pred, loss_type)`

### Training (4 functions)
10. `create_batches(X, y, batch_size, shuffle)`
11. `train_epoch(network, X, y, batch_size, optimizer)`
12. `train(network, X_train, y_train, X_val, y_val, ...)`
13. `early_stopping(val_losses, patience)`

## Mathematical Background

### Dense Layer Forward
```
z = Wx + b
a = activation(z)

Dimensions:
  x: (batch_size, n_input)
  W: (n_input, n_output)
  b: (n_output,)
  z: (batch_size, n_output)
```

### Dense Layer Backward
```
Given: da (gradient from next layer)

If activation was used:
  dz = da * activation'(z)  # element-wise
Else:
  dz = da

Gradients:
  dW = x^T @ dz / batch_size
  db = mean(dz, axis=0)
  dx = dz @ W^T  # pass to previous layer
```

### Loss Functions
```
MSE:
  L = (1/n) Σ (y - ŷ)²
  ∂L/∂ŷ = (2/n)(ŷ - y)

Cross-Entropy:
  L = -(1/n) Σ y log(ŷ)
  ∂L/∂z = ŷ - y  (for softmax + CE)
```

## Common Pitfalls

### 1. Forgetting to Cache Values
```python
# Need x for backward pass!
def forward(self, x):
    self.x = x  # CACHE!
    return self.activation(x @ self.W + self.b)
```

### 2. Shape Mismatches
```python
# Always check shapes!
print(f"x: {x.shape}, W: {self.W.shape}, result: {result.shape}")
```

### 3. Gradient Accumulation
```python
# Reset gradients before each batch
self.grad_W = np.zeros_like(self.W)
```

## Success Criteria

- Gradients match numerical differentiation
- XOR problem solved with 2-layer network
- Training loss decreases
- Network generalizes to test data

**Target: All tests passing**
