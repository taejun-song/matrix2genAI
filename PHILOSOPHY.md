# Matrix2GenAI: Learning Philosophy

## Core Principle: Simple Building Blocks

This curriculum teaches AI/ML through **simple, composable building blocks** rather than complete frameworks. Students learn by implementing fundamental primitives and composing them into complex systems.

## Why Building Blocks?

### ❌ What We DON'T Do
```python
# Bad: Complete framework that hides the magic
model = LinearRegression()
model.fit(X, y)  # What happens inside?
predictions = model.predict(X_test)
```

### ✅ What We DO
```python
# Good: Simple functions that students compose
def predict(X, weights, bias):
    return X @ weights + bias

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(X, y_true, y_pred):
    errors = y_pred - y_true
    grad_w = (2/len(X)) * X.T @ errors
    grad_b = (2/len(X)) * np.sum(errors)
    return grad_w, grad_b

# Students build their own training loop
for epoch in range(num_epochs):
    y_pred = predict(X, weights, bias)
    loss = mse_loss(y, y_pred)
    grad_w, grad_b = mse_gradient(X, y, y_pred)
    weights -= learning_rate * grad_w
    bias -= learning_rate * grad_b
```

## Key Principles

### 1. Each Function Does One Thing
- `sigmoid(x)` - just computes sigmoid
- `sigmoid_derivative(x)` - just computes derivative
- `cross_entropy_loss(y, y_pred)` - just computes loss
- No hidden state, no magic

### 2. No Classes (Unless Necessary)
- Functions > Classes for building blocks
- Classes only when state management is essential (e.g., computational graphs, optimizers with momentum)
- Prefer pure functions that are easy to test and compose

### 3. Explicit > Implicit
```python
# Bad: Hidden state and magic
layer.forward(X)  # Where does output go?
layer.backward()  # Where does gradient come from?

# Good: Explicit inputs and outputs
output = dense_forward(X, weights, bias)
grad_X, grad_w, grad_b = dense_backward(grad_output, X, weights)
```

### 4. Students Write the Loops
We provide:
- Loss functions
- Gradient functions
- Forward/backward pass functions

Students write:
- Training loops
- Validation loops
- The actual learning algorithm

### 5. Composability
Each building block should work with others:

```python
# Building blocks
z1 = dense_forward(X, W1, b1)
a1 = relu(z1)
z2 = dense_forward(a1, W2, b2)
y_pred = sigmoid(z2)
loss = binary_cross_entropy(y, y_pred)

# Student composes them into a 2-layer network
# Then adds backprop by calling backward functions in reverse
```

## Examples by Stage

### s06: Linear Regression Building Blocks

**Simple Functions:**
```python
# Core primitives
def linear_predict(X, weights, bias):
    """Just compute Xw + b"""
    return X @ weights + bias

def mse_loss(y_true, y_pred):
    """Just compute mean squared error"""
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(X, y, y_pred):
    """Compute gradients w.r.t weights and bias"""
    n = len(y)
    errors = y_pred - y
    grad_w = (2/n) * X.T @ errors
    grad_b = (2/n) * np.sum(errors)
    return grad_w, grad_b

def normal_equation(X, y):
    """Closed-form solution"""
    X_aug = np.column_stack([np.ones(len(X)), X])
    w_aug = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    return w_aug[1:], w_aug[0]  # weights, bias
```

**Student Assignment:**
"Implement these 4 functions, then write a training loop that:
1. Uses gradient descent to minimize MSE
2. Plots loss over epochs
3. Compares with normal equation solution"

### s12: Neural Network Building Blocks

**Simple Functions:**
```python
# Layer operations
def dense_forward(X, W, b):
    """Forward pass through dense layer"""
    return X @ W + b

def dense_backward(grad_output, X, W):
    """Backward pass through dense layer"""
    grad_X = grad_output @ W.T
    grad_W = X.T @ grad_output
    grad_b = np.sum(grad_output, axis=0)
    return grad_X, grad_W, grad_b

# Activations
def relu(x):
    return np.maximum(0, x)

def relu_backward(grad_output, x):
    return grad_output * (x > 0)

# Loss
def softmax_cross_entropy_loss(logits, labels):
    """Combined softmax + cross entropy (numerically stable)"""
    # ... implementation

def softmax_cross_entropy_backward(logits, labels):
    """Gradient of softmax + cross entropy"""
    # ... implementation
```

**Student Assignment:**
"Use these building blocks to:
1. Implement a 2-layer MLP forward pass
2. Implement backpropagation using the backward functions
3. Train on MNIST-like data
4. Experiment with different architectures by composing blocks"

### s13: CNN Building Blocks

**Simple Functions:**
```python
def conv2d_forward(X, kernel, bias, stride=1, padding=0):
    """2D convolution forward pass"""
    # Implementation
    return output

def conv2d_backward(grad_output, X, kernel, stride=1, padding=0):
    """2D convolution backward pass"""
    # Returns grad_X, grad_kernel, grad_bias
    return grad_X, grad_kernel, grad_bias

def maxpool2d_forward(X, pool_size, stride):
    """Max pooling forward pass"""
    return output, mask  # mask for backward pass

def maxpool2d_backward(grad_output, mask):
    """Max pooling backward pass"""
    return grad_X
```

**Student Assignment:**
"Build a simple CNN by composing:
- conv2d → relu → maxpool2d → flatten → dense → softmax
- Implement forward and backward passes
- Train on image classification"

### s27: LLM Building Blocks

**Simple Functions:**
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute attention output"""
    # Implementation
    return output, attention_weights

def attention_backward(grad_output, Q, K, V, attention_weights):
    """Backward pass through attention"""
    return grad_Q, grad_K, grad_V

def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization"""
    # Implementation
    return output, cache  # cache for backward

def layer_norm_backward(grad_output, cache):
    """Layer norm backward pass"""
    return grad_x, grad_gamma, grad_beta

def gpt_block_forward(x, params):
    """One transformer decoder block"""
    # Compose: attention → add&norm → ffn → add&norm
    return output, cache

def gpt_block_backward(grad_output, cache):
    """Backward through GPT block"""
    return grad_x, grad_params
```

**Student Assignment:**
"Build a mini-GPT by:
1. Stacking multiple gpt_blocks
2. Adding embedding and output layers
3. Implementing next-token prediction training
4. Generating text with different sampling strategies"

## Testing Philosophy

Each building block gets:

### 1. Unit Tests (Correctness)
```python
def test_relu():
    x = np.array([-1, 0, 1, 2])
    expected = np.array([0, 0, 1, 2])
    np.testing.assert_array_equal(relu(x), expected)
```

### 2. Gradient Tests (Numerical Checking)
```python
def test_relu_gradient():
    x = np.random.randn(10)
    grad_output = np.ones_like(x)

    # Numerical gradient
    numerical = compute_numerical_gradient(relu, x)

    # Analytical gradient
    analytical = relu_backward(grad_output, x)

    np.testing.assert_allclose(numerical, analytical, atol=1e-5)
```

### 3. Composition Tests (Integration)
```python
def test_two_layer_network():
    """Test composing dense → relu → dense → sigmoid"""
    # Forward
    z1 = dense_forward(X, W1, b1)
    a1 = relu(z1)
    z2 = dense_forward(a1, W2, b2)
    output = sigmoid(z2)

    # Backward (student implements this part)
    # ...
```

### 4. Reference Tests (Validation)
```python
def test_conv2d_vs_pytorch():
    """Compare with PyTorch implementation"""
    import torch.nn.functional as F

    our_output = conv2d_forward(X, kernel, bias, stride, padding)
    torch_output = F.conv2d(torch_X, torch_kernel, torch_bias, stride, padding)

    np.testing.assert_allclose(our_output, torch_output.numpy(), atol=1e-5)
```

## Progression Strategy

### Stage Structure

Each stage teaches 5-10 building blocks:

```
stages/sXX_topic_name/
├── README.md           # Concepts and learning objectives
├── spec.md             # Function specifications
├── starter/
│   ├── primitives.py   # TODO: Implement building blocks
│   └── utils.py        # Optional: helper functions (provided)
└── tests/
    ├── test_primitives.py      # Test each building block
    └── test_composition.py     # Test composing blocks
```

### Example: s12 (Neural Networks)

**Student implements:**
- `dense_forward(X, W, b)` - 5 lines
- `dense_backward(grad, X, W)` - 5 lines
- `relu(x)` - 1 line
- `relu_backward(grad, x)` - 1 line
- `sigmoid(x)` - 1 line
- `sigmoid_backward(grad, x)` - 2 lines
- `softmax_cross_entropy_loss(logits, labels)` - 10 lines
- `softmax_cross_entropy_backward(logits, labels)` - 5 lines

**Then composes them into:**
- 2-layer network with training loop (20-30 lines of composition code)
- Tests each block individually
- Tests composition

### Complexity Growth

- **s01-s05**: Pure math functions (dot product, matrix multiply, derivatives)
- **s06-s10**: ML building blocks (loss functions, metrics, optimizers)
- **s11-s15**: Neural net primitives (layers, activations, backprop)
- **s16-s19**: Advanced components (attention, transformers)
- **s20-s23**: RL primitives (value functions, policy updates)
- **s24-s28**: Generative components (VAE encoder/decoder, GAN, diffusion step)

## Benefits of This Approach

### For Students:
1. **Deep understanding** - Know exactly what each piece does
2. **Debuggability** - Easy to inspect intermediate values
3. **Flexibility** - Can modify any part
4. **Transferable** - Understanding carries to PyTorch/TensorFlow
5. **Testable** - Each function has clear inputs/outputs

### For Teaching:
1. **Modular** - Can teach/test each concept independently
2. **Gradual** - Add complexity by composing simple pieces
3. **Clear** - No hidden magic or framework complexity
4. **Practical** - Students write real training loops, not just fill TODOs

### For Research:
1. **Hackable** - Easy to try new ideas at any level
2. **Understandable** - Can trace through entire computation
3. **Minimal** - No framework overhead or assumptions

## Anti-Patterns to Avoid

### ❌ Over-Abstraction
```python
class Layer(ABC):
    @abstractmethod
    def forward(self, x): pass

    @abstractmethod
    def backward(self, grad): pass

class Dense(Layer):
    def forward(self, x):
        self.x = x  # Hidden state!
        return x @ self.W + self.b
```

**Why bad:** Hides what's happening, requires understanding OOP, manages state

### ❌ Framework Features
```python
def fit(self, X, y, validation_data=None, callbacks=None,
        verbose=1, batch_size=32, epochs=100, ...):
    # 100 lines of framework code
```

**Why bad:** Students don't learn the core algorithm, just API usage

### ❌ Magic Numbers
```python
def initialize_weights(layer_type, size):
    if layer_type == "conv":
        return np.random.randn(*size) * np.sqrt(2/size[0])  # He init
    elif layer_type == "dense":
        return np.random.randn(*size) * np.sqrt(1/size[0])  # Xavier
```

**Why bad:** Hides important concepts behind magic strings

### ✅ Simple and Clear
```python
def he_initialization(shape, fan_in):
    """He initialization for ReLU networks.

    Returns weights ~ N(0, sqrt(2/fan_in))
    Good for ReLU activations.
    """
    return np.random.randn(*shape) * np.sqrt(2 / fan_in)

def xavier_initialization(shape, fan_in):
    """Xavier initialization for tanh/sigmoid networks.

    Returns weights ~ N(0, sqrt(1/fan_in))
    Good for tanh/sigmoid activations.
    """
    return np.random.randn(*shape) * np.sqrt(1 / fan_in)
```

**Why good:** Clear purpose, explicit choice, easy to understand

## Summary

**We teach by providing simple, composable primitives that students combine into complete systems.**

- ✅ Simple functions over complex classes
- ✅ Explicit operations over hidden magic
- ✅ Student-written loops over framework methods
- ✅ Composition over inheritance
- ✅ Pure functions over stateful objects
- ✅ Clear building blocks over abstraction layers

**Result:** Students deeply understand ML/AI from first principles and can build anything.
