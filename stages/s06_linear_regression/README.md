# Stage 6: Linear Regression - Building Blocks

## Overview

Build a complete linear regression system from simple, composable functions. No classes, no frameworks - just pure functions that you'll compose into a working ML system.

**The Big Idea:** Linear regression finds the "best fit line" through data points. It's the foundation of all machine learning - once you understand this, everything else is just variations!

## Learning Philosophy

You will implement **small building blocks** (1-3 lines each) that compose into complete ML algorithms:
- Gradient descent training loop
- Visualization and evaluation
- Comparison with closed-form solution

**Time:** 2-3 hours
**Difficulty:** ⭐⭐⭐

## Getting Started

### Setup

```bash
# Navigate to this stage
cd stages/s06_linear_regression

# Run tests (using uv - recommended)
uv run pytest tests/ -v

# Or activate venv first
source .venv/bin/activate  # On Unix/macOS
# .venv\Scripts\activate   # On Windows
pytest tests/ -v
```

### Files You'll Edit

- `starter/primitives.py` - Implement the 8 building blocks here
- Tests are in `tests/test_primitives.py` - run them to verify your work

### Workflow

1. Read this README to understand concepts
2. Implement sub-functions in `starter/primitives.py` (follow TODOs)
3. Run `uv run pytest tests/test_primitives.py::TestClassName -v` to test each function
4. Once all tests pass, compose functions into training loops
5. Experiment and visualize!

---

## Conceptual Understanding

### What is Linear Regression?

**Goal:** Given data points, find the line that best fits them.

```
Example: Predict house price from size

Data points:          Best fit line:
  Size | Price         Price = 150·Size + 50000
  1000 | 200k
  1500 | 275k          y
  2000 | 350k          |     ╱ (prediction line)
                       |   ╱
Question:              | ╱
  Size=1800 → Price=?  |______ x (size)

Answer: 150·1800 + 50000 = $320k
```

**The Math:** ŷ = w·x + b
- **w** (weight/slope): How much price increases per sq ft
- **b** (bias/intercept): Base price (where line crosses y-axis)
- **ŷ**: Predicted value

### Why Do We Need a Loss Function?

**Problem:** Infinite possible lines! Which one is "best"?

```
Three possible lines:
  Line A: ŷ = 100x + 70000   (too flat, underestimates)
  Line B: ŷ = 150x + 50000   (just right!)
  Line C: ŷ = 200x + 30000   (too steep, overestimates)

How do we measure "goodness of fit"?
```

**Solution:** Mean Squared Error (MSE)
```
MSE = (1/n) Σ (actual - predicted)²

Line A: MSE = 5000  (bad)
Line B: MSE = 100   (good!)
Line C: MSE = 3000  (bad)

Why square the errors?
  • Penalizes big mistakes more (10² = 100 vs 2·10 = 20)
  • No negative errors canceling positive ones
  • Smooth, differentiable (can use calculus!)
```

### Gradient Descent: The Optimization Algorithm

**Intuition:** Imagine you're blindfolded on a hill, trying to reach the valley (minimum MSE).

```
Strategy:
  1. Feel which direction is downhill (compute gradient)
  2. Take a small step in that direction (update weights)
  3. Repeat until you reach the valley (convergence)

In math:
  gradient = ∂MSE/∂w = how much MSE changes when we change w

  Update rule:
    w_new = w_old - learning_rate · gradient

  Example:
    Current: w=100, MSE=5000
    Gradient: ∂MSE/∂w = +500 (increasing w increases MSE)
    Update: w = 100 - 0.01·500 = 95 (move opposite direction!)
```

### The R² Score: Measuring Success

**Question:** Is MSE=100 good or bad? Depends on scale!

**R²:** Compares your model to the simplest baseline (predicting mean)

```
R² = 1 - (your_model_error / baseline_error)

Interpretation:
  R² = 1.0  → Perfect! Every prediction exactly right
  R² = 0.9  → Explains 90% of variance (very good)
  R² = 0.5  → Explains 50% of variance (okay)
  R² = 0.0  → As good as predicting average (bad)
  R² < 0.0  → Worse than predicting average (terrible!)

Example:
  Baseline (predict mean): always predict $250k
    Error on test set: SS_tot = 10,000

  Your model: ŷ = 150x + 50000
    Error on test set: SS_res = 1,000

  R² = 1 - 1000/10000 = 0.9  ✓ Great!
```

### Normal Equations: The Shortcut

**Two ways to find the best line:**

```
1. Gradient Descent (iterative):
   ✓ Works for any model
   ✓ Memory efficient
   ✗ Needs hyperparameters (learning rate)
   ✗ May take many iterations

2. Normal Equations (closed-form):
   ✓ Direct solution (one step!)
   ✓ Always finds optimal answer
   ✗ Only works for linear regression
   ✗ Slow for large datasets (matrix inversion)

Formula: w = (XᵀX)⁻¹Xᵀy

When to use:
  • Small dataset (< 10,000 samples): Normal equations
  • Large dataset (> 10,000 samples): Gradient descent
```

## What You'll Build

### Core Building Blocks (8 functions)

1. `predict(X, weights, bias)` - Compute predictions
2. `mse_loss(y_true, y_pred)` - Mean squared error
3. `mse_gradient(X, y_true, y_pred)` - Loss gradients
4. `r2_score(y_true, y_pred)` - Coefficient of determination
5. `normal_equation(X, y)` - Closed-form solution
6. `standardize(X)` - Z-score normalization
7. `train_test_split(X, y, test_size)` - Data splitting
8. `polynomial_features(X, degree)` - Feature expansion

### What You'll Compose

Using these blocks, you'll write:
- A gradient descent training loop
- Model evaluation pipeline
- Experiments comparing different approaches

## Mathematical Background

### Linear Model
```
ŷ = Xw + b

where:
  X ∈ ℝⁿˣᵈ  (n samples, d features)
  w ∈ ℝᵈ    (weights)
  b ∈ ℝ     (bias)
  ŷ ∈ ℝⁿ    (predictions)
```

### Mean Squared Error (MSE)
```
L(w,b) = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```

### Gradients
```
∂L/∂w = (2/n) Σᵢ (ŷᵢ - yᵢ) · xᵢ = (2/n) Xᵀ(ŷ - y)
∂L/∂b = (2/n) Σᵢ (ŷᵢ - yᵢ)
```

### Gradient Descent Update
```
w ← w - α · ∂L/∂w
b ← b - α · ∂L/∂b
```

### Normal Equations (Closed-Form)
```
[b]   = ([1ₙ | X]ᵀ [1ₙ | X])⁻¹ [1ₙ | X]ᵀ y
[w]
```

### R² Score
```
R² = 1 - (SS_res / SS_tot)

where:
  SS_res = Σ(y - ŷ)²  (residual sum of squares)
  SS_tot = Σ(y - ȳ)²  (total sum of squares)
```

## Implementation Guide

### Step 1: Core Primitives (30 min)

Implement the 8 building blocks. Each is simple:

**Example: `predict`**
```python
def predict(X, weights, bias):
    return X @ weights + bias  # That's it!
```

**Example: `mse_loss`**
```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)  # One line!
```

### Step 2: Test Each Block (30 min)

Run tests to verify each function works:
```bash
# If using uv (recommended):
uv run pytest tests/test_primitives.py -v

# Or if venv is activated:
pytest tests/test_primitives.py -v
```

Each function is tested independently with clear inputs/outputs.

### Step 3: Compose Training Loop (45 min)

Write your own gradient descent loop:

```python
# Initialize
weights = np.zeros(n_features)
bias = 0.0
learning_rate = 0.01

# Train
for epoch in range(1000):
    # Forward pass
    y_pred = predict(X_train, weights, bias)

    # Compute loss
    loss = mse_loss(y_train, y_pred)

    # Compute gradients
    grad_w, grad_b = mse_gradient(X_train, y_train, y_pred)

    # Update parameters
    weights -= learning_rate * grad_w
    bias -= learning_rate * grad_b

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

### Step 4: Evaluate and Visualize (30 min)

Use your building blocks:

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize
X_train_scaled, mean, std = standardize(X_train)
X_test_scaled = (X_test - mean) / std

# Train with gradient descent
# ... (your training loop)

# Compare with normal equation
weights_ne, bias_ne = normal_equation(X_train_scaled, y_train)

# Evaluate both
y_pred_gd = predict(X_test_scaled, weights, bias)
y_pred_ne = predict(X_test_scaled, weights_ne, bias_ne)

print(f"GD R²: {r2_score(y_test, y_pred_gd):.4f}")
print(f"NE R²: {r2_score(y_test, y_pred_ne):.4f}")
```

## Key Concepts

### Why Standardization Matters

Without standardization:
```
Feature 1: [1, 2, 3]          → gradient scale ~1
Feature 2: [1000, 2000, 3000] → gradient scale ~1000
```

Gradient descent struggles! After standardization:
```
Feature 1: [-1, 0, 1]  → gradient scale ~1
Feature 2: [-1, 0, 1]  → gradient scale ~1
```

Much better convergence!

### Gradient Descent vs Normal Equations

**Gradient Descent:**
- ✅ Works with huge datasets
- ✅ Can do online learning
- ✅ Generalizes to non-linear models
- ❌ Requires tuning learning rate
- ❌ Needs multiple iterations

**Normal Equations:**
- ✅ Exact solution in one step
- ✅ No hyperparameters
- ❌ Requires matrix inversion O(d³)
- ❌ Needs full dataset in memory
- ❌ Doesn't work for non-linear models

### R² Interpretation

- **R² = 1.0**: Perfect predictions
- **R² = 0.0**: Model is as good as predicting the mean
- **R² < 0.0**: Model is worse than predicting the mean!

## Common Pitfalls

### 1. Forgetting to Standardize

```python
# ❌ Bad: Different scales
X_train = np.array([[1, 1000], [2, 2000]])
# Gradient descent will struggle!

# ✅ Good: Same scale
X_train_scaled, mean, std = standardize(X_train)
# [[−1, −1], [1, 1]] - much better!
```

### 2. Not Applying Same Transform to Test Data

```python
# ❌ Bad: Different transforms
X_train_scaled, mean, std = standardize(X_train)
X_test_scaled, _, _ = standardize(X_test)  # Different mean/std!

# ✅ Good: Use training statistics
X_train_scaled, mean, std = standardize(X_train)
X_test_scaled = (X_test - mean) / std  # Same transform
```

### 3. Learning Rate Too Large

```python
learning_rate = 1.0  # Too large!
# Loss will oscillate or diverge: 100 → 50 → 200 → 1000 → NaN

learning_rate = 0.01  # Better
# Loss decreases smoothly: 100 → 50 → 25 → 12 → 6 → 3
```

### 4. Not Checking Loss Decreases

```python
# Always monitor loss!
for epoch in range(1000):
    loss = mse_loss(y_train, y_pred)

    # Sanity check
    if epoch > 0 and loss > prev_loss:
        print(f"Warning: Loss increased! {prev_loss:.4f} → {loss:.4f}")
        print("Learning rate might be too large!")

    prev_loss = loss
```

## Experiments to Try

Once your building blocks work, experiment:

### 1. Learning Rate Comparison
```python
for lr in [0.001, 0.01, 0.1, 1.0]:
    # Train model with this learning rate
    # Plot loss curves
    # Which converges fastest?
```

### 2. Polynomial Features
```python
# Try different polynomial degrees
for degree in [1, 2, 3, 5]:
    X_poly = polynomial_features(X, degree)
    # Does higher degree always improve R²?
    # What about test R² vs train R²?
```

### 3. Dataset Size Effect
```python
for n in [10, 50, 100, 500, 1000]:
    X_subset = X[:n]
    y_subset = y[:n]
    # How does R² change with more data?
    # When does normal equation become slow?
```

## Debugging Guide

### Predictions are all wrong
```python
# Check your prediction function
X = np.array([[1, 2], [3, 4]])
w = np.array([1, 1])
b = 0

y_pred = predict(X, w, b)
print(y_pred)  # Should be [3, 7]

# If not, check matrix shapes:
print(f"X shape: {X.shape}")      # (2, 2)
print(f"w shape: {w.shape}")      # (2,)
print(f"X @ w shape: {(X @ w).shape}")  # (2,)
```

### Loss not decreasing
```python
# 1. Check gradient computation
from stages.s03_calculus.starter.numerical_diff import gradient_check

def loss_fn(w):
    y_pred = predict(X, w, bias)
    return mse_loss(y, y_pred)

# Compare analytical vs numerical gradients
grad_analytical, _ = mse_gradient(X, y, y_pred)
is_correct = gradient_check(loss_fn, weights, grad_analytical)
print(f"Gradient correct: {is_correct}")

# 2. Try smaller learning rate
learning_rate /= 10

# 3. Check if standardization is applied
print(f"X mean: {X.mean(axis=0)}")  # Should be ~0 if standardized
print(f"X std: {X.std(axis=0)}")    # Should be ~1 if standardized
```

### Normal equation fails
```python
# Check for singular matrix (correlated features)
X_aug = np.column_stack([np.ones(len(X)), X])
rank = np.linalg.matrix_rank(X_aug)
print(f"Matrix rank: {rank}, Expected: {X_aug.shape[1]}")

if rank < X_aug.shape[1]:
    print("Matrix is singular! Features might be correlated.")
    # Use np.linalg.lstsq instead of solve
```

## Testing Your Implementation

```bash
# Test individual building blocks
pytest tests/test_primitives.py -v

# Test gradient correctness
pytest tests/test_gradients.py -v

# Test full composition
pytest tests/test_composition.py -v

# Grade your work
python scripts/grade.py s06_linear_regression
```

## Real-World Applications

Linear regression is everywhere:

- **Housing Prices**: Predict price from sqft, bedrooms, location
- **Sales Forecasting**: Predict revenue from advertising spend
- **Medical**: Predict patient outcomes from measurements
- **Finance**: Model stock returns from economic indicators
- **Science**: Discover relationships between variables

Even in 2024, linear regression is often the **first thing** ML engineers try because:
- Fast to train
- Easy to interpret (feature coefficients)
- Strong baseline for comparison
- Works surprisingly well for many problems

## What's Next

After mastering these building blocks:

**s07: Logistic Regression** - Add sigmoid and cross-entropy loss
**s08: Feature Engineering** - PCA, encoding, imputation
**s09: Regularization** - L1/L2 penalties (just add to gradient!)
**s12: Neural Networks** - Stack linear layers with non-linearities

The pattern stays the same:
1. Simple building block functions
2. Test each independently
3. Compose into complete systems

## Success Criteria

You understand this stage when you can:

- ✅ Explain what each of the 8 functions does
- ✅ Write a training loop from scratch
- ✅ Debug why loss isn't decreasing
- ✅ Explain when to use GD vs normal equations
- ✅ Add new features (e.g., momentum) by modifying the loop

**Target: 90%+ test passing rate**

Good luck! Remember: each function is simple. The power comes from composition. 🚀
