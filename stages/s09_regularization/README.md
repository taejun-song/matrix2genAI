# Stage 9: Regularization & Model Selection

## Overview

Build regularized regression models and model selection tools from scratch. Learn how to prevent overfitting and find optimal hyperparameters.

**The Big Idea:** Regularization adds a penalty term to the loss function that discourages overly complex models. Combined with cross-validation, it's your defense against overfitting!

## Learning Philosophy

You will implement **small building blocks** (1-5 lines each) that compose into complete systems:
- Ridge, Lasso, and ElasticNet regularization
- K-fold cross-validation
- Grid search for hyperparameter tuning
- Learning and validation curves

**Time:** 4-5 hours
**Difficulty:** ⭐⭐⭐

## Getting Started

### Setup

```bash
cd stages/s09_regularization

uv run pytest tests/ -v

source .venv/bin/activate  # On Unix/macOS
pytest tests/ -v
```

### Files You'll Edit

- `starter/regularization.py` - Regularization penalties and gradients
- `starter/cross_validation.py` - K-fold and stratified splitting
- `starter/model_selection.py` - Grid search and learning curves
- Tests in `tests/` verify your implementations

### Workflow

1. Read conceptual sections below to understand WHY
2. Implement sub-functions (follow TODOs)
3. Test each module: `uv run pytest tests/test_regularization.py -v`
4. Compose functions into model selection pipelines

---

## Conceptual Understanding

### The Overfitting Problem

When a model fits training data too well, it memorizes noise instead of learning patterns:

```
Underfitting (High Bias):
  Training Error: High
  Test Error: High
  Model too simple - can't capture patterns

Just Right:
  Training Error: Low
  Test Error: Low
  Model captures true patterns

Overfitting (High Variance):
  Training Error: Very Low (nearly 0!)
  Test Error: High
  Model memorizes training data, including noise
```

### How Regularization Helps

Regularization adds a penalty for large weights:

```
Without Regularization:
  Loss = MSE(predictions, targets)
  Problem: Model can make weights arbitrarily large to fit noise

With Regularization:
  Loss = MSE(predictions, targets) + λ × penalty(weights)

  The penalty term discourages large weights!
  - λ = 0: No regularization (can overfit)
  - λ = ∞: All weights forced to 0 (underfits)
  - λ optimal: Balance between fit and simplicity
```

### Ridge vs Lasso vs ElasticNet

**Ridge (L2): Square the weights**
```
Penalty = α × Σᵢ wᵢ²

Properties:
- Shrinks all weights toward zero
- Never sets weights exactly to zero
- Good when all features are relevant
- Handles correlated features well
```

**Lasso (L1): Absolute value of weights**
```
Penalty = α × Σᵢ |wᵢ|

Properties:
- Can set weights exactly to zero (feature selection!)
- Creates sparse models
- Picks one feature from correlated groups
- Harder to optimize (not differentiable at 0)
```

**ElasticNet: Best of both**
```
Penalty = α × [ρ × Σᵢ |wᵢ| + (1-ρ)/2 × Σᵢ wᵢ²]

Properties:
- ρ = 1: Pure Lasso
- ρ = 0: Pure Ridge
- ρ = 0.5: Equal mix
- Good for many correlated features
```

### Visualizing the Constraint Regions

```
Ridge (L2):                     Lasso (L1):
        w₂                              w₂
         |                               |
    .----|----.                     /\   |
   /     |     \                   /  \  |
  |      |      |                 /    \ |
--+------+------+--w₁       ----/------\+------w₁
  |      |      |               \      /|
   \     |     /                 \    / |
    '----|----'                   \  /  |
         |                         \/   |

  Circle: no corners              Diamond: corners at axes!
  Weights shrink uniformly        Weights hit zero at corners
```

The Lasso diamond has corners on the axes - that's why it produces zeros!

### Why Cross-Validation?

**Problem:** We can't use test data for model selection (that's cheating!)

**Solution:** K-fold cross-validation

```
Original Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

5-Fold CV:
  Fold 1: Train=[3..10], Val=[1,2]    → Score₁
  Fold 2: Train=[1,2,5..10], Val=[3,4] → Score₂
  Fold 3: Train=[1..4,7..10], Val=[5,6] → Score₃
  Fold 4: Train=[1..6,9,10], Val=[7,8] → Score₄
  Fold 5: Train=[1..8], Val=[9,10]     → Score₅

Final Score = mean([Score₁, Score₂, ..., Score₅])

Every sample is used for validation exactly once!
```

## What You'll Build

### Regularization Functions (6)

1. `ridge_penalty(weights, alpha)` - L2 penalty term
2. `ridge_gradient(weights, alpha)` - L2 gradient
3. `lasso_penalty(weights, alpha)` - L1 penalty term
4. `lasso_subgradient(weights, alpha)` - L1 subgradient
5. `elastic_net_penalty(weights, alpha, l1_ratio)` - Combined penalty
6. `elastic_net_gradient(weights, alpha, l1_ratio)` - Combined gradient

### Cross-Validation Functions (4)

7. `create_folds(n_samples, n_folds)` - Generate fold indices
8. `cross_val_score(model_fn, X, y, n_folds)` - Compute CV scores
9. `stratified_folds(y, n_folds)` - Stratified split for classification
10. `cross_val_predict(model_fn, X, y, n_folds)` - Get predictions for all samples

### Model Selection Functions (4)

11. `grid_search(model_fn, param_grid, X, y, n_folds)` - Find best hyperparameters
12. `learning_curve(model_fn, X, y, train_sizes)` - Score vs training size
13. `validation_curve(model_fn, X, y, param_name, param_range)` - Score vs hyperparameter
14. `best_alpha_ridge(X, y, alphas, n_folds)` - Find optimal Ridge alpha

## Mathematical Background

### Ridge Regression (L2)

```
Loss(w) = (1/2n) Σᵢ (yᵢ - ŷᵢ)² + α × (1/2) Σⱼ wⱼ²

Penalty: P(w) = (α/2) × ||w||₂² = (α/2) × Σⱼ wⱼ²

Gradient of penalty:
  ∂P/∂wⱼ = α × wⱼ

Full gradient:
  ∂Loss/∂w = (1/n) × Xᵀ(ŷ - y) + α × w
```

### Lasso Regression (L1)

```
Loss(w) = (1/2n) Σᵢ (yᵢ - ŷᵢ)² + α × Σⱼ |wⱼ|

Penalty: P(w) = α × ||w||₁ = α × Σⱼ |wⱼ|

Subgradient of |w|:
  ∂|w|/∂w = sign(w) = { +1 if w > 0
                      { -1 if w < 0
                      { [-1, +1] if w = 0  (set of valid subgradients)

For optimization, use:
  sign(w) where sign(0) = 0
```

### ElasticNet

```
Loss(w) = (1/2n) Σᵢ (yᵢ - ŷᵢ)² + α × [ρ × ||w||₁ + (1-ρ)/2 × ||w||₂²]

Parameters:
  α (alpha): Overall regularization strength
  ρ (l1_ratio): Mix between L1 and L2
    - ρ = 1.0: Pure Lasso
    - ρ = 0.0: Pure Ridge
    - ρ = 0.5: Equal contribution

Gradient:
  ∂Loss/∂w = MSE_gradient + α × [ρ × sign(w) + (1-ρ) × w]
```

## Implementation Guide

### Step 1: Regularization Penalties (15 min)

```python
def ridge_penalty(weights, alpha):
    return (alpha / 2) * np.sum(weights ** 2)

def ridge_gradient(weights, alpha):
    return alpha * weights

def lasso_penalty(weights, alpha):
    return alpha * np.sum(np.abs(weights))

def lasso_subgradient(weights, alpha):
    return alpha * np.sign(weights)
```

### Step 2: Cross-Validation (25 min)

```python
def create_folds(n_samples, n_folds):
    indices = np.arange(n_samples)
    fold_sizes = np.full(n_folds, n_samples // n_folds)
    fold_sizes[:n_samples % n_folds] += 1

    folds = []
    current = 0
    for size in fold_sizes:
        val_idx = indices[current:current + size]
        train_idx = np.concatenate([indices[:current], indices[current + size:]])
        folds.append((train_idx, val_idx))
        current += size
    return folds
```

### Step 3: Grid Search (20 min)

```python
def grid_search(model_fn, param_grid, X, y, n_folds=5):
    best_score = -np.inf
    best_params = None

    for params in param_combinations(param_grid):
        scores = cross_val_score(
            lambda X_tr, y_tr: model_fn(X_tr, y_tr, **params),
            X, y, n_folds
        )
        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return {'best_params': best_params, 'best_score': best_score}
```

### Step 4: Learning Curves (15 min)

```python
def learning_curve(model_fn, X, y, train_sizes):
    train_scores = []
    val_scores = []

    for size in train_sizes:
        X_subset, y_subset = X[:size], y[:size]
        scores = cross_val_score(model_fn, X_subset, y_subset)
        # ... compute train and val scores

    return train_sizes, train_scores, val_scores
```

## Key Concepts

### Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Noise

High Regularization (large α):
  - High bias (underfitting)
  - Low variance
  - Model too simple

Low Regularization (small α):
  - Low bias
  - High variance (overfitting)
  - Model too complex

Optimal α:
  - Balance between bias and variance
  - Found via cross-validation
```

### Regularization Path

The "regularization path" shows how weights change with α:

```
Weight
  ^
  |  *****
  |       ***
  |          **
  |            *
  |             *-------- (Ridge: smooth decay)
  |              \
  |               *
  |                \_____ (Lasso: hits zero!)
  +-----------------------> α (regularization strength)
  0   0.1   0.5   1.0
```

### Coordinate Descent for Lasso

Since L1 isn't differentiable at 0, we use coordinate descent:

```python
for j in range(n_features):
    # Compute optimal w[j] while keeping others fixed
    residual = y - X @ w + X[:, j] * w[j]
    rho = X[:, j] @ residual

    # Soft thresholding!
    if rho < -alpha:
        w[j] = (rho + alpha) / (X[:, j] @ X[:, j])
    elif rho > alpha:
        w[j] = (rho - alpha) / (X[:, j] @ X[:, j])
    else:
        w[j] = 0  # Lasso sets to exactly zero!
```

## Common Pitfalls

### 1. Forgetting to Standardize

```python
# Ridge/Lasso penalty treats all features equally
X = np.array([[1, 1000], [2, 2000]])

# Without standardization:
# Feature 2 dominates because it's larger!
# Regularization penalizes w₂ more unfairly

# Always standardize first!
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
```

### 2. Including Bias in Penalty

```python
# DON'T penalize the bias term!
penalty = alpha * np.sum(weights ** 2)  # Only weights, not bias!

# The bias (intercept) should be free to shift predictions
# Penalizing it would force predictions toward 0
```

### 3. Using Test Data for Model Selection

```python
# WRONG: Data leakage!
for alpha in alphas:
    model = train(X_train, y_train, alpha)
    score = evaluate(model, X_test, y_test)  # Cheating!
    if score > best_score:
        best_alpha = alpha

# CORRECT: Use cross-validation on training data only
best_alpha = grid_search_cv(X_train, y_train, alphas)
final_model = train(X_train, y_train, best_alpha)
final_score = evaluate(final_model, X_test, y_test)  # Only once!
```

### 4. Wrong Number of Folds

```python
# Too few folds: High variance in estimate
# Too many folds: Computationally expensive, high variance

# Rule of thumb:
# n_folds = 5 or 10 for most cases
# n_folds = n (leave-one-out) only for tiny datasets
```

## Debugging Guide

### Regularization Not Helping

```python
# Check if alpha is too small
print(f"Alpha: {alpha}")
print(f"MSE: {mse_loss}")
print(f"Penalty: {ridge_penalty(weights, alpha)}")

# If penalty << MSE, try larger alpha
# Typical alphas: 0.001, 0.01, 0.1, 1, 10, 100
```

### Cross-Validation Scores Vary Wildly

```python
scores = cross_val_score(model_fn, X, y, n_folds=5)
print(f"Scores: {scores}")
print(f"Mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# High std indicates:
# - Not enough data
# - Model too complex
# - Data has different distributions
```

### Lasso Selects No Features

```python
# Alpha too large - all weights become zero
print(f"Non-zero weights: {np.sum(weights != 0)}")

# Try smaller alpha
# Use cross-validation to find optimal alpha
```

## Experiments to Try

### 1. Compare Ridge vs Lasso on Correlated Features

```python
# Create correlated features
X1 = np.random.randn(100)
X2 = X1 + 0.1 * np.random.randn(100)  # X2 ≈ X1
X = np.column_stack([X1, X2])
y = 2 * X1 + np.random.randn(100) * 0.1

# Train Ridge and Lasso
# Ridge: Both weights ≈ 1
# Lasso: One weight ≈ 2, other ≈ 0
```

### 2. Regularization Path Visualization

```python
alphas = np.logspace(-4, 2, 50)
weights_path = []

for alpha in alphas:
    w = train_ridge(X, y, alpha)
    weights_path.append(w)

# Plot weights vs alpha
# Watch how they shrink toward zero
```

### 3. Learning Curve Analysis

```python
train_sizes = [10, 50, 100, 200, 500]
train_scores, val_scores = learning_curve(model, X, y, train_sizes)

# If gap between train and val is large: Overfitting
# If both are low: Underfitting
```

## Testing Your Implementation

```bash
pytest tests/test_regularization.py -v

pytest tests/test_cross_validation.py -v

pytest tests/test_model_selection.py -v

python scripts/grade.py s09_regularization
```

## Real-World Applications

Regularization is essential in practice:

- **Gene Expression Analysis**: Thousands of genes, few samples → Lasso selects relevant genes
- **Financial Modeling**: Many correlated indicators → ElasticNet for stability
- **Image Classification**: High-dimensional features → Ridge prevents overfitting
- **NLP**: Sparse features (bag of words) → Lasso for feature selection
- **Recommendation Systems**: Regularized matrix factorization

## What's Next

After mastering regularization:

**s10: Decision Trees** - Non-linear models with built-in feature selection
**s11: Perceptrons** - Neural network building blocks
**s12: Neural Networks** - Regularization via dropout, weight decay

The concepts stay the same:
1. Simple penalty functions
2. Cross-validation for hyperparameters
3. Bias-variance tradeoff

## Success Criteria

You understand this stage when you can:

- Explain the difference between Ridge and Lasso
- Derive the gradient of L2 penalty
- Implement K-fold cross-validation from scratch
- Use grid search to find optimal hyperparameters
- Interpret learning curves to diagnose overfitting
- Know when to use each type of regularization

**Target: All tests passing**

Good luck! Regularization is one of the most important techniques in machine learning.
