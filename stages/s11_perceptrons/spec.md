# Stage 11: Perceptrons & Activation Functions - Specification

## Building Blocks to Implement

You will implement **20 functions** organized into 3 modules:
1. **Activations** (8 functions) - Activation functions
2. **Activation Gradients** (8 functions) - Derivatives
3. **Perceptron** (4 functions) - Perceptron algorithm

---

## Module 1: Activations

### `relu(z) → activated`

Rectified Linear Unit: f(z) = max(0, z)

**Args:**
- `z`: Input values, any shape

**Returns:**
- `activated`: Same shape as z

**Formula:**
```
f(z) = max(0, z) = { z  if z > 0
                   { 0  if z ≤ 0
```

**Example:**
```python
relu(np.array([-2, -1, 0, 1, 2]))
# array([0, 0, 0, 1, 2])
```

**Implementation:**
```python
def relu(z: np.ndarray) -> np.ndarray:
    # TODO: return np.maximum(0, z)
    raise NotImplementedError
```

---

### `leaky_relu(z, alpha) → activated`

Leaky ReLU: allows small negative values.

**Args:**
- `z`: Input values
- `alpha`: Slope for negative values (default 0.01)

**Formula:**
```
f(z) = { z     if z > 0
       { αz    if z ≤ 0
```

**Example:**
```python
leaky_relu(np.array([-2, -1, 0, 1, 2]), alpha=0.1)
# array([-0.2, -0.1, 0, 1, 2])
```

**Implementation:**
```python
def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    # TODO: return np.where(z > 0, z, alpha * z)
    raise NotImplementedError
```

---

### `elu(z, alpha) → activated`

Exponential Linear Unit: smooth for negative values.

**Args:**
- `z`: Input values
- `alpha`: Scale for negative region (default 1.0)

**Formula:**
```
f(z) = { z           if z > 0
       { α(e^z - 1)  if z ≤ 0
```

**Example:**
```python
elu(np.array([-2, 0, 2]), alpha=1.0)
# array([-0.865, 0, 2])
```

**Implementation:**
```python
def elu(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    # TODO: return np.where(z > 0, z, alpha * (np.exp(z) - 1))
    raise NotImplementedError
```

---

### `gelu(z) → activated`

Gaussian Error Linear Unit: used in transformers.

**Args:**
- `z`: Input values

**Formula (approximation):**
```
f(z) ≈ 0.5 × z × (1 + tanh(√(2/π) × (z + 0.044715z³)))
```

**Example:**
```python
gelu(np.array([-1, 0, 1]))
# array([-0.159, 0, 0.841])
```

**Implementation:**
```python
def gelu(z: np.ndarray) -> np.ndarray:
    # TODO:
    # c = np.sqrt(2 / np.pi)
    # return 0.5 * z * (1 + np.tanh(c * (z + 0.044715 * z**3)))
    raise NotImplementedError
```

---

### `swish(z, beta) → activated`

Swish / SiLU: self-gated activation.

**Args:**
- `z`: Input values
- `beta`: Scaling parameter (default 1.0)

**Formula:**
```
f(z) = z × σ(βz)  where σ is sigmoid
```

**Example:**
```python
swish(np.array([-1, 0, 1]), beta=1.0)
# array([-0.269, 0, 0.731])
```

**Implementation:**
```python
def swish(z: np.ndarray, beta: float = 1.0) -> np.ndarray:
    # TODO:
    # sigmoid = 1 / (1 + np.exp(-beta * z))
    # return z * sigmoid
    raise NotImplementedError
```

---

### `tanh_activation(z) → activated`

Hyperbolic tangent: outputs in (-1, 1).

**Args:**
- `z`: Input values

**Formula:**
```
f(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

**Example:**
```python
tanh_activation(np.array([-1, 0, 1]))
# array([-0.762, 0, 0.762])
```

**Implementation:**
```python
def tanh_activation(z: np.ndarray) -> np.ndarray:
    # TODO: return np.tanh(z)
    raise NotImplementedError
```

---

### `softplus(z) → activated`

Softplus: smooth approximation of ReLU.

**Args:**
- `z`: Input values

**Formula:**
```
f(z) = log(1 + e^z)
```

**Numerical Stability:**
For large z, use f(z) ≈ z to avoid overflow.

**Example:**
```python
softplus(np.array([-1, 0, 1]))
# array([0.313, 0.693, 1.313])
```

**Implementation:**
```python
def softplus(z: np.ndarray) -> np.ndarray:
    # TODO: Use stable computation
    # return np.where(z > 20, z, np.log1p(np.exp(z)))
    raise NotImplementedError
```

---

### `mish(z) → activated`

Mish: self-regularized activation.

**Args:**
- `z`: Input values

**Formula:**
```
f(z) = z × tanh(softplus(z)) = z × tanh(log(1 + e^z))
```

**Example:**
```python
mish(np.array([-1, 0, 1]))
# array([-0.303, 0, 0.865])
```

**Implementation:**
```python
def mish(z: np.ndarray) -> np.ndarray:
    # TODO: return z * np.tanh(softplus(z))
    raise NotImplementedError
```

---

## Module 2: Activation Gradients

### `relu_derivative(z) → gradient`

Derivative of ReLU.

**Formula:**
```
f'(z) = { 1  if z > 0
        { 0  if z ≤ 0
```

**Example:**
```python
relu_derivative(np.array([-1, 0, 1]))
# array([0, 0, 1])
```

**Implementation:**
```python
def relu_derivative(z: np.ndarray) -> np.ndarray:
    # TODO: return (z > 0).astype(float)
    raise NotImplementedError
```

---

### `leaky_relu_derivative(z, alpha) → gradient`

Derivative of Leaky ReLU.

**Formula:**
```
f'(z) = { 1  if z > 0
        { α  if z ≤ 0
```

**Implementation:**
```python
def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    # TODO: return np.where(z > 0, 1.0, alpha)
    raise NotImplementedError
```

---

### `elu_derivative(z, alpha) → gradient`

Derivative of ELU.

**Formula:**
```
f'(z) = { 1          if z > 0
        { f(z) + α   if z ≤ 0
```

**Implementation:**
```python
def elu_derivative(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    # TODO: return np.where(z > 0, 1.0, elu(z, alpha) + alpha)
    raise NotImplementedError
```

---

### `gelu_derivative(z) → gradient`

Derivative of GELU (approximate).

**Formula:**
```
f'(z) ≈ 0.5(1 + tanh(c(z + 0.044715z³))) +
        0.5z × sech²(c(z + 0.044715z³)) × c(1 + 3×0.044715z²)

where c = √(2/π)
```

**Implementation:**
```python
def gelu_derivative(z: np.ndarray) -> np.ndarray:
    # TODO: Implement approximate derivative
    # c = np.sqrt(2 / np.pi)
    # inner = c * (z + 0.044715 * z**3)
    # tanh_inner = np.tanh(inner)
    # sech2 = 1 - tanh_inner**2
    # return 0.5 * (1 + tanh_inner) + 0.5 * z * sech2 * c * (1 + 3 * 0.044715 * z**2)
    raise NotImplementedError
```

---

### `swish_derivative(z, beta) → gradient`

Derivative of Swish.

**Formula:**
```
f'(z) = σ(βz) + βz × σ(βz) × (1 - σ(βz))
      = σ(βz) × (1 + βz × (1 - σ(βz)))
```

**Implementation:**
```python
def swish_derivative(z: np.ndarray, beta: float = 1.0) -> np.ndarray:
    # TODO:
    # sig = 1 / (1 + np.exp(-beta * z))
    # return sig * (1 + beta * z * (1 - sig))
    raise NotImplementedError
```

---

### `tanh_derivative(z) → gradient`

Derivative of tanh.

**Formula:**
```
f'(z) = 1 - tanh²(z)
```

**Implementation:**
```python
def tanh_derivative(z: np.ndarray) -> np.ndarray:
    # TODO: return 1 - np.tanh(z)**2
    raise NotImplementedError
```

---

### `softplus_derivative(z) → gradient`

Derivative of softplus (equals sigmoid!).

**Formula:**
```
f'(z) = 1 / (1 + e^(-z)) = σ(z)
```

**Implementation:**
```python
def softplus_derivative(z: np.ndarray) -> np.ndarray:
    # TODO: return 1 / (1 + np.exp(-z))
    raise NotImplementedError
```

---

### `mish_derivative(z) → gradient`

Derivative of Mish.

**Formula:**
```
Let sp = softplus(z), ω = 4(z+1) + 4e^(2z) + e^(3z) + e^z(4z+6)
    δ = 2e^z + e^(2z) + 2

f'(z) = e^z × ω / δ²
```

**Implementation:**
```python
def mish_derivative(z: np.ndarray) -> np.ndarray:
    # TODO: Implement using chain rule
    # sp = softplus(z)
    # tanh_sp = np.tanh(sp)
    # sech2_sp = 1 - tanh_sp**2
    # sp_deriv = softplus_derivative(z)
    # return tanh_sp + z * sech2_sp * sp_deriv
    raise NotImplementedError
```

---

## Module 3: Perceptron

### `perceptron_forward(x, weights, bias) → prediction`

Single perceptron forward pass.

**Args:**
- `x`: Single input vector, shape (n_features,)
- `weights`: Weight vector, shape (n_features,)
- `bias`: Bias term (scalar)

**Returns:**
- `prediction`: 0 or 1

**Formula:**
```
z = w·x + b
ŷ = 1 if z > 0 else 0
```

**Implementation:**
```python
def perceptron_forward(
    x: np.ndarray, weights: np.ndarray, bias: float
) -> int:
    # TODO:
    # z = np.dot(x, weights) + bias
    # return 1 if z > 0 else 0
    raise NotImplementedError
```

---

### `perceptron_update(x, y, weights, bias, lr) → (new_weights, new_bias)`

Single perceptron weight update.

**Args:**
- `x`: Input vector
- `y`: True label (0 or 1)
- `weights`: Current weights
- `bias`: Current bias
- `lr`: Learning rate

**Returns:**
- `new_weights`: Updated weights
- `new_bias`: Updated bias

**Algorithm:**
```
1. Compute prediction ŷ
2. If ŷ ≠ y:
   - weights ← weights + lr × (y - ŷ) × x
   - bias ← bias + lr × (y - ŷ)
```

**Implementation:**
```python
def perceptron_update(
    x: np.ndarray,
    y: int,
    weights: np.ndarray,
    bias: float,
    lr: float = 1.0,
) -> tuple[np.ndarray, float]:
    # TODO:
    # y_pred = perceptron_forward(x, weights, bias)
    # if y_pred != y:
    #     weights = weights + lr * (y - y_pred) * x
    #     bias = bias + lr * (y - y_pred)
    # return weights, bias
    raise NotImplementedError
```

---

### `perceptron_train(X, y, lr, n_epochs) → (weights, bias, history)`

Train perceptron on dataset.

**Args:**
- `X`: Features, shape (n_samples, n_features)
- `y`: Labels (0 or 1), shape (n_samples,)
- `lr`: Learning rate
- `n_epochs`: Number of epochs

**Returns:**
- `weights`: Trained weights
- `bias`: Trained bias
- `history`: List of errors per epoch

**Implementation:**
```python
def perceptron_train(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 1.0,
    n_epochs: int = 100,
) -> tuple[np.ndarray, float, list[int]]:
    # TODO:
    # n_features = X.shape[1]
    # weights = np.zeros(n_features)
    # bias = 0.0
    # history = []
    #
    # for epoch in range(n_epochs):
    #     errors = 0
    #     for i in range(len(X)):
    #         y_pred = perceptron_forward(X[i], weights, bias)
    #         if y_pred != y[i]:
    #             errors += 1
    #         weights, bias = perceptron_update(X[i], y[i], weights, bias, lr)
    #     history.append(errors)
    #     if errors == 0:
    #         break
    #
    # return weights, bias, history
    raise NotImplementedError
```

---

### `perceptron_predict(X, weights, bias) → predictions`

Batch predictions.

**Args:**
- `X`: Features, shape (n_samples, n_features)
- `weights`: Trained weights
- `bias`: Trained bias

**Returns:**
- `predictions`: Array of 0/1, shape (n_samples,)

**Implementation:**
```python
def perceptron_predict(
    X: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    # TODO:
    # z = X @ weights + bias
    # return (z > 0).astype(int)
    raise NotImplementedError
```

---

## Testing

```bash
pytest stages/s11_perceptrons/tests/ -v
python scripts/grade.py s11_perceptrons
```

---

## Success Criteria

- All activations match expected values
- Derivatives pass numerical gradient check
- Perceptron converges on AND/OR gates
- Perceptron fails to converge on XOR (as expected!)
