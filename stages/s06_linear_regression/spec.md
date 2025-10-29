# Stage 6: Linear Regression - Specification

## Building Blocks to Implement

You will implement **8 simple functions**. Each function is a pure building block with clear inputs and outputs.

---

## 1. Prediction

### `predict(X, weights, bias) → predictions`

Compute linear predictions: ŷ = Xw + b

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `weights`: Weight vector, shape `(n_features,)`
- `bias`: Bias term (scalar)

**Returns:**
- `predictions`: shape `(n_samples,)`

**Formula:**
```
ŷᵢ = Σⱼ Xᵢⱼ · wⱼ + b
```

**Example:**
```python
X = np.array([[1, 2], [3, 4], [5, 6]])
weights = np.array([0.5, 1.0])
bias = 0.1

predictions = predict(X, weights, bias)
# [2.6, 5.6, 8.6]  # (1*0.5 + 2*1.0 + 0.1), etc.
```

**Implementation:**
```python
def predict(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Compute linear predictions.

    Args:
        X: Input features, shape (n_samples, n_features)
        weights: Weight vector, shape (n_features,)
        bias: Bias term (scalar)

    Returns:
        predictions: Predicted values, shape (n_samples,)
    """
    # TODO: One line - use @ for matrix multiplication
    pass
```

---

## 2. Loss Function

### `mse_loss(y_true, y_pred) → loss`

Compute mean squared error.

**Args:**
- `y_true`: True values, shape `(n_samples,)`
- `y_pred`: Predicted values, shape `(n_samples,)`

**Returns:**
- `loss`: Scalar value

**Formula:**
```
MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```

**Example:**
```python
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 2.2, 2.8])

loss = mse_loss(y_true, y_pred)
# 0.03  # Mean of [0.01, 0.04, 0.04]
```

**Implementation:**
```python
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean squared error.

    Args:
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)

    Returns:
        mse: Mean squared error (scalar)
    """
    # TODO: One line - use np.mean()
    pass
```

---

## 3. Gradient Computation

### `mse_gradient(X, y_true, y_pred) → (grad_w, grad_b)`

Compute gradients of MSE loss w.r.t. weights and bias.

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `y_true`: True values, shape `(n_samples,)`
- `y_pred`: Predicted values, shape `(n_samples,)`

**Returns:**
- `grad_w`: Gradient w.r.t. weights, shape `(n_features,)`
- `grad_b`: Gradient w.r.t. bias (scalar)

**Formulas:**
```
errors = ŷ - y

∂L/∂w = (2/n) · Xᵀ · errors
∂L/∂b = (2/n) · Σᵢ errorsᵢ
```

**Derivation:**
```
L = (1/n) Σᵢ (yᵢ - ŷᵢ)²
ŷᵢ = Σⱼ Xᵢⱼwⱼ + b

∂L/∂wⱼ = (1/n) Σᵢ 2(ŷᵢ - yᵢ) · ∂ŷᵢ/∂wⱼ
       = (1/n) Σᵢ 2(ŷᵢ - yᵢ) · Xᵢⱼ
       = (2/n) Σᵢ (ŷᵢ - yᵢ) · Xᵢⱼ
       = (2/n) Xᵀ(ŷ - y)

∂L/∂b  = (2/n) Σᵢ (ŷᵢ - yᵢ)
```

**Example:**
```python
X = np.array([[1, 2], [3, 4]])
y_true = np.array([5, 11])
y_pred = np.array([5.5, 11.5])

grad_w, grad_b = mse_gradient(X, y_true, y_pred)
# grad_w = [2.0, 3.0]  # (2/2) * X.T @ [0.5, 0.5]
# grad_b = 0.5         # (2/2) * sum([0.5, 0.5])
```

**Implementation:**
```python
def mse_gradient(
    X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Compute gradients of MSE loss.

    Args:
        X: Input features, shape (n_samples, n_features)
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)

    Returns:
        grad_w: Gradient w.r.t. weights, shape (n_features,)
        grad_b: Gradient w.r.t. bias (scalar)
    """
    n = len(y_true)
    errors = y_pred - y_true

    # TODO: Compute gradients using formulas above
    # grad_w: Use X.T @ errors
    # grad_b: Use np.sum(errors)

    pass
```

---

## 4. Evaluation Metric

### `r2_score(y_true, y_pred) → r2`

Compute R² (coefficient of determination).

**Args:**
- `y_true`: True values, shape `(n_samples,)`
- `y_pred`: Predicted values, shape `(n_samples,)`

**Returns:**
- `r2`: R² score (scalar), range: (-∞, 1]

**Formula:**
```
SS_res = Σᵢ (yᵢ - ŷᵢ)²   (residual sum of squares)
SS_tot = Σᵢ (yᵢ - ȳ)²    (total sum of squares)

R² = 1 - SS_res / SS_tot
```

**Interpretation:**
- R² = 1.0: Perfect predictions
- R² = 0.0: Model as good as predicting mean
- R² < 0.0: Model worse than predicting mean

**Edge case:**
- If `SS_tot == 0` (all y values are the same), return `0.0`

**Example:**
```python
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])

r2 = r2_score(y_true, y_pred)
# ~0.99  # Very good fit
```

**Implementation:**
```python
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² score.

    Args:
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)

    Returns:
        r2: R² score (scalar)
    """
    # TODO: Implement R² formula
    # 1. Compute SS_res = sum of squared residuals
    # 2. Compute SS_tot = sum of squared deviations from mean
    # 3. Handle edge case where SS_tot == 0
    # 4. Return 1 - SS_res/SS_tot

    pass
```

---

## 5. Closed-Form Solution

### `normal_equation(X, y) → (weights, bias)`

Solve for optimal weights using normal equations (analytical solution).

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `y`: Target values, shape `(n_samples,)`

**Returns:**
- `weights`: Optimal weights, shape `(n_features,)`
- `bias`: Optimal bias (scalar)

**Formula:**
```
Augment X: X_aug = [1ₙ | X]  (prepend column of ones)

Solve: w_aug = (X_augᵀ X_aug)⁻¹ X_augᵀ y

Extract: bias = w_aug[0], weights = w_aug[1:]
```

**Implementation Notes:**
- Use `np.linalg.lstsq()` instead of computing inverse explicitly
- This is more numerically stable and handles singular matrices

**Example:**
```python
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])

weights, bias = normal_equation(X, y)
# weights = [2.0], bias ≈ 0.0  # Perfect fit: y = 2x
```

**Implementation:**
```python
def normal_equation(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Solve linear regression using normal equations.

    Args:
        X: Input features, shape (n_samples, n_features)
        y: Target values, shape (n_samples,)

    Returns:
        weights: Optimal weights, shape (n_features,)
        bias: Optimal bias (scalar)
    """
    # TODO: Implement normal equations
    # 1. Create X_aug by prepending column of ones: np.column_stack([np.ones(n), X])
    # 2. Solve using np.linalg.lstsq(X_aug, y, rcond=None)[0]
    # 3. Extract bias (first element) and weights (rest)

    pass
```

---

## 6. Feature Scaling

### `standardize(X) → (X_scaled, mean, std)`

Standardize features to zero mean and unit variance (Z-score normalization).

**Args:**
- `X`: Features, shape `(n_samples, n_features)`

**Returns:**
- `X_scaled`: Standardized features, shape `(n_samples, n_features)`
- `mean`: Mean of each feature, shape `(n_features,)`
- `std`: Standard deviation of each feature, shape `(n_features,)`

**Formula:**
```
X_scaled = (X - μ) / σ

where:
  μ = mean(X, axis=0)
  σ = std(X, axis=0)
```

**Edge Case:**
- If `std` is zero for any feature (constant feature), use `std = 1` to avoid division by zero

**Why This Matters:**
Gradient descent converges much faster when features have similar scales.

**Example:**
```python
X = np.array([[1, 100], [2, 200], [3, 300]])

X_scaled, mean, std = standardize(X)
# X_scaled ≈ [[-1, -1], [0, 0], [1, 1]]
# mean = [2, 200]
# std = [0.816, 81.6]
```

**Implementation:**
```python
def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using Z-score normalization.

    Args:
        X: Input features, shape (n_samples, n_features)

    Returns:
        X_scaled: Standardized features, shape (n_samples, n_features)
        mean: Mean of each feature, shape (n_features,)
        std: Standard deviation of each feature, shape (n_features,)
    """
    # TODO: Implement standardization
    # 1. Compute mean: np.mean(X, axis=0)
    # 2. Compute std: np.std(X, axis=0)
    # 3. Handle zero std: std = np.where(std == 0, 1, std)
    # 4. Scale: X_scaled = (X - mean) / std

    pass
```

---

## 7. Data Splitting

### `train_test_split(X, y, test_size) → (X_train, X_test, y_train, y_test)`

Split data into training and test sets.

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `y`: Targets, shape `(n_samples,)`
- `test_size`: Fraction for test set, range (0, 1)

**Returns:**
- `X_train`: Training features
- `X_test`: Test features
- `y_train`: Training targets
- `y_test`: Test targets

**Implementation:**
- Split at index: `split_idx = int(len(X) * (1 - test_size))`
- No shuffling (keep it simple for now)

**Example:**
```python
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_train = [[1], [2], [3], [4]]  # 80%
# X_test = [[5]]                   # 20%
```

**Implementation:**
```python
def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.

    Args:
        X: Features, shape (n_samples, n_features)
        y: Targets, shape (n_samples,)
        test_size: Fraction of data for testing

    Returns:
        X_train, X_test, y_train, y_test
    """
    # TODO: Implement train/test split
    # 1. Compute split index
    # 2. Slice arrays: X_train = X[:split_idx]

    pass
```

---

## 8. Feature Engineering

### `polynomial_features(X, degree) → X_poly`

Generate polynomial features up to specified degree.

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `degree`: Maximum polynomial degree

**Returns:**
- `X_poly`: Polynomial features, shape `(n_samples, n_poly_features)`

**Example:**
```python
X = np.array([[2], [3]])
X_poly = polynomial_features(X, degree=3)

# X_poly = [[2, 4, 8],     # [x, x², x³]
#           [3, 9, 27]]    # [x, x², x³]
```

**For Multiple Features:**
```python
X = np.array([[a, b]])
degree = 2

# X_poly = [[a, b, a², ab, b²]]
```

**Implementation:**
Use `itertools.combinations_with_replacement` to generate all combinations.

**Implementation:**
```python
def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Generate polynomial features.

    Args:
        X: Input features, shape (n_samples, n_features)
        degree: Maximum polynomial degree

    Returns:
        X_poly: Polynomial features, shape (n_samples, n_poly_features)
    """
    # TODO: Implement polynomial feature generation
    # HINT: Use itertools.combinations_with_replacement
    # Example structure:
    #
    # from itertools import combinations_with_replacement
    # features = [X]  # Start with degree 1
    #
    # for deg in range(2, degree + 1):
    #     for combo in combinations_with_replacement(range(n_features), deg):
    #         product = X[:, combo[0]]
    #         for idx in combo[1:]:
    #             product = product * X[:, idx]
    #         features.append(product.reshape(-1, 1))
    #
    # return np.hstack(features)

    pass
```

---

## Composition Example

Once you implement these 8 functions, you can compose them:

```python
# Load data
X, y = load_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize
X_train_scaled, mean, std = standardize(X_train)
X_test_scaled = (X_test - mean) / std

# Initialize parameters
n_features = X_train_scaled.shape[1]
weights = np.zeros(n_features)
bias = 0.0

# Training loop (gradient descent)
learning_rate = 0.01
num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass
    y_pred = predict(X_train_scaled, weights, bias)

    # Compute loss
    loss = mse_loss(y_train, y_pred)

    # Compute gradients
    grad_w, grad_b = mse_gradient(X_train_scaled, y_train, y_pred)

    # Update parameters
    weights -= learning_rate * grad_w
    bias -= learning_rate * grad_b

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

# Evaluate on test set
y_test_pred = predict(X_test_scaled, weights, bias)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Test R²: {test_r2:.4f}")

# Compare with normal equation
weights_ne, bias_ne = normal_equation(X_train_scaled, y_train)
y_test_pred_ne = predict(X_test_scaled, weights_ne, bias_ne)
test_r2_ne = r2_score(y_test, y_test_pred_ne)
print(f"Normal Equation R²: {test_r2_ne:.4f}")
```

---

## Constraints

### Allowed
- NumPy: All functions
- Python standard library
- `itertools` for polynomial features

### Not Allowed
- Scikit-learn (except for comparison in tests)
- SciPy
- Any ML frameworks

### Code Style
- Pure functions (no side effects)
- Type hints for all functions
- Clear, simple implementations
- No classes needed!

---

## Testing

Your implementation will be tested on:

1. **Correctness**: Each function produces correct outputs
2. **Gradient Accuracy**: Gradients match numerical differentiation
3. **Edge Cases**: Empty arrays, single samples, zero variance
4. **Composition**: Functions work together correctly
5. **Performance**: MSE decreases during training

Run tests:
```bash
pytest stages/s06_linear_regression/tests/ -v
python scripts/grade.py s06_linear_regression
```

---

## Success Criteria

✅ All 8 functions pass unit tests
✅ Gradients match numerical differentiation (< 1e-5 error)
✅ Training loop decreases loss
✅ R² > 0.7 on test data (for typical datasets)
✅ Normal equation matches gradient descent solution (< 1e-3 difference)

Good luck! Each function is simple - the magic is in how they compose. 🚀
