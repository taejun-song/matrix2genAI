# Stage 9: Regularization & Model Selection - Specification

## Building Blocks to Implement

You will implement **14 functions** organized into 3 modules:
1. **Regularization** (6 functions) - Penalty terms and gradients
2. **Cross-Validation** (4 functions) - Data splitting strategies
3. **Model Selection** (4 functions) - Hyperparameter tuning

---

## Module 1: Regularization

### `ridge_penalty(weights, alpha) → penalty`

Compute L2 regularization penalty: (α/2) × ||w||₂²

**Args:**
- `weights`: Weight vector, shape `(n_features,)`
- `alpha`: Regularization strength (scalar, α ≥ 0)

**Returns:**
- `penalty`: Scalar value

**Formula:**
```
P(w) = (α/2) × Σⱼ wⱼ² = (α/2) × wᵀw
```

**Example:**
```python
weights = np.array([1.0, 2.0, 3.0])
alpha = 0.1

penalty = ridge_penalty(weights, alpha)
# 0.7  # (0.1/2) * (1 + 4 + 9) = 0.05 * 14 = 0.7
```

**Implementation:**
```python
def ridge_penalty(weights: np.ndarray, alpha: float) -> float:
    """
    Compute L2 (Ridge) regularization penalty.

    Args:
        weights: Weight vector, shape (n_features,)
        alpha: Regularization strength

    Returns:
        penalty: L2 penalty value
    """
    # TODO: return (alpha / 2) * np.sum(weights ** 2)
    raise NotImplementedError
```

---

### `ridge_gradient(weights, alpha) → gradient`

Compute gradient of L2 penalty w.r.t. weights.

**Args:**
- `weights`: Weight vector, shape `(n_features,)`
- `alpha`: Regularization strength

**Returns:**
- `gradient`: Gradient vector, shape `(n_features,)`

**Formula:**
```
∂P/∂w = α × w
```

**Example:**
```python
weights = np.array([1.0, 2.0, 3.0])
alpha = 0.1

gradient = ridge_gradient(weights, alpha)
# [0.1, 0.2, 0.3]
```

**Implementation:**
```python
def ridge_gradient(weights: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute gradient of L2 penalty.

    Args:
        weights: Weight vector, shape (n_features,)
        alpha: Regularization strength

    Returns:
        gradient: Gradient of L2 penalty, shape (n_features,)
    """
    # TODO: return alpha * weights
    raise NotImplementedError
```

---

### `lasso_penalty(weights, alpha) → penalty`

Compute L1 regularization penalty: α × ||w||₁

**Args:**
- `weights`: Weight vector, shape `(n_features,)`
- `alpha`: Regularization strength

**Returns:**
- `penalty`: Scalar value

**Formula:**
```
P(w) = α × Σⱼ |wⱼ| = α × ||w||₁
```

**Example:**
```python
weights = np.array([1.0, -2.0, 3.0])
alpha = 0.1

penalty = lasso_penalty(weights, alpha)
# 0.6  # 0.1 * (1 + 2 + 3) = 0.6
```

**Implementation:**
```python
def lasso_penalty(weights: np.ndarray, alpha: float) -> float:
    """
    Compute L1 (Lasso) regularization penalty.

    Args:
        weights: Weight vector, shape (n_features,)
        alpha: Regularization strength

    Returns:
        penalty: L1 penalty value
    """
    # TODO: return alpha * np.sum(np.abs(weights))
    raise NotImplementedError
```

---

### `lasso_subgradient(weights, alpha) → subgradient`

Compute subgradient of L1 penalty w.r.t. weights.

**Args:**
- `weights`: Weight vector, shape `(n_features,)`
- `alpha`: Regularization strength

**Returns:**
- `subgradient`: Subgradient vector, shape `(n_features,)`

**Formula:**
```
∂P/∂wⱼ = α × sign(wⱼ)

where sign(x) = { +1 if x > 0
               { -1 if x < 0
               {  0 if x = 0
```

**Note:** At wⱼ = 0, the L1 norm is not differentiable. We use sign(0) = 0 as a valid subgradient.

**Example:**
```python
weights = np.array([1.0, -2.0, 0.0])
alpha = 0.1

subgradient = lasso_subgradient(weights, alpha)
# [0.1, -0.1, 0.0]
```

**Implementation:**
```python
def lasso_subgradient(weights: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute subgradient of L1 penalty.

    Args:
        weights: Weight vector, shape (n_features,)
        alpha: Regularization strength

    Returns:
        subgradient: Subgradient of L1 penalty, shape (n_features,)
    """
    # TODO: return alpha * np.sign(weights)
    raise NotImplementedError
```

---

### `elastic_net_penalty(weights, alpha, l1_ratio) → penalty`

Compute ElasticNet penalty: α × [ρ × ||w||₁ + (1-ρ)/2 × ||w||₂²]

**Args:**
- `weights`: Weight vector, shape `(n_features,)`
- `alpha`: Overall regularization strength
- `l1_ratio`: Mix between L1 and L2 (ρ ∈ [0, 1])

**Returns:**
- `penalty`: Scalar value

**Formula:**
```
P(w) = α × [ρ × ||w||₁ + (1-ρ)/2 × ||w||₂²]

Special cases:
  ρ = 1.0: Pure Lasso
  ρ = 0.0: Pure Ridge
```

**Example:**
```python
weights = np.array([1.0, 2.0])
alpha = 1.0
l1_ratio = 0.5

penalty = elastic_net_penalty(weights, alpha, l1_ratio)
# 0.5 * 3 + 0.25 * 5 = 1.5 + 1.25 = 2.75
```

**Implementation:**
```python
def elastic_net_penalty(
    weights: np.ndarray, alpha: float, l1_ratio: float
) -> float:
    """
    Compute ElasticNet regularization penalty.

    Args:
        weights: Weight vector, shape (n_features,)
        alpha: Overall regularization strength
        l1_ratio: Mix ratio (1.0 = pure L1, 0.0 = pure L2)

    Returns:
        penalty: ElasticNet penalty value
    """
    # TODO:
    # l1_penalty = np.sum(np.abs(weights))
    # l2_penalty = np.sum(weights ** 2)
    # return alpha * (l1_ratio * l1_penalty + (1 - l1_ratio) / 2 * l2_penalty)
    raise NotImplementedError
```

---

### `elastic_net_gradient(weights, alpha, l1_ratio) → gradient`

Compute gradient of ElasticNet penalty.

**Args:**
- `weights`: Weight vector, shape `(n_features,)`
- `alpha`: Overall regularization strength
- `l1_ratio`: Mix between L1 and L2

**Returns:**
- `gradient`: Gradient vector, shape `(n_features,)`

**Formula:**
```
∂P/∂w = α × [ρ × sign(w) + (1-ρ) × w]
```

**Example:**
```python
weights = np.array([1.0, -2.0])
alpha = 1.0
l1_ratio = 0.5

gradient = elastic_net_gradient(weights, alpha, l1_ratio)
# [0.5 * 1 + 0.5 * 1, 0.5 * (-1) + 0.5 * (-2)] = [1.0, -1.5]
```

**Implementation:**
```python
def elastic_net_gradient(
    weights: np.ndarray, alpha: float, l1_ratio: float
) -> np.ndarray:
    """
    Compute gradient of ElasticNet penalty.

    Args:
        weights: Weight vector, shape (n_features,)
        alpha: Overall regularization strength
        l1_ratio: Mix ratio

    Returns:
        gradient: Gradient of ElasticNet penalty, shape (n_features,)
    """
    # TODO:
    # l1_grad = np.sign(weights)
    # l2_grad = weights
    # return alpha * (l1_ratio * l1_grad + (1 - l1_ratio) * l2_grad)
    raise NotImplementedError
```

---

## Module 2: Cross-Validation

### `create_folds(n_samples, n_folds) → folds`

Create indices for K-fold cross-validation.

**Args:**
- `n_samples`: Total number of samples
- `n_folds`: Number of folds (K)

**Returns:**
- `folds`: List of tuples `(train_indices, val_indices)`

**Algorithm:**
```
1. Compute fold sizes (handle non-divisible cases)
2. For each fold:
   - Validation: samples in current fold
   - Training: all other samples
```

**Example:**
```python
folds = create_folds(n_samples=10, n_folds=5)
# [
#   (array([2..9]), array([0,1])),   # Fold 0
#   (array([0,1,4..9]), array([2,3])), # Fold 1
#   ...
# ]
```

**Implementation:**
```python
def create_folds(n_samples: int, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create K-fold cross-validation splits.

    Args:
        n_samples: Total number of samples
        n_folds: Number of folds

    Returns:
        folds: List of (train_indices, val_indices) tuples
    """
    # TODO:
    # indices = np.arange(n_samples)
    # fold_sizes = np.full(n_folds, n_samples // n_folds)
    # fold_sizes[:n_samples % n_folds] += 1
    #
    # folds = []
    # current = 0
    # for size in fold_sizes:
    #     val_idx = indices[current:current + size]
    #     train_idx = np.concatenate([indices[:current], indices[current + size:]])
    #     folds.append((train_idx, val_idx))
    #     current += size
    # return folds
    raise NotImplementedError
```

---

### `cross_val_score(model_fn, X, y, n_folds, score_fn) → scores`

Compute cross-validation scores.

**Args:**
- `model_fn`: Function `(X_train, y_train) → model` that trains and returns a model
- `X`: Features, shape `(n_samples, n_features)`
- `y`: Targets, shape `(n_samples,)`
- `n_folds`: Number of folds
- `score_fn`: Function `(y_true, y_pred) → score` for evaluation

**Returns:**
- `scores`: Array of scores, shape `(n_folds,)`

**Algorithm:**
```
1. Create folds
2. For each fold:
   a. Split data into train/val
   b. Train model on train data
   c. Predict on val data
   d. Compute score
3. Return all scores
```

**Example:**
```python
def train_linear(X_train, y_train):
    # Returns a prediction function
    w, b = normal_equation(X_train, y_train)
    return lambda X: X @ w + b

scores = cross_val_score(train_linear, X, y, n_folds=5, score_fn=r2_score)
# array([0.85, 0.87, 0.82, 0.89, 0.84])
```

**Implementation:**
```python
def cross_val_score(
    model_fn: callable,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    score_fn: callable = None,
) -> np.ndarray:
    """
    Compute cross-validation scores.

    Args:
        model_fn: Function that takes (X_train, y_train) and returns a predictor
        X: Features, shape (n_samples, n_features)
        y: Targets, shape (n_samples,)
        n_folds: Number of CV folds
        score_fn: Scoring function (y_true, y_pred) -> score

    Returns:
        scores: Array of validation scores, shape (n_folds,)
    """
    # TODO:
    # folds = create_folds(len(X), n_folds)
    # scores = []
    # for train_idx, val_idx in folds:
    #     X_train, X_val = X[train_idx], X[val_idx]
    #     y_train, y_val = y[train_idx], y[val_idx]
    #     model = model_fn(X_train, y_train)
    #     y_pred = model(X_val)
    #     scores.append(score_fn(y_val, y_pred))
    # return np.array(scores)
    raise NotImplementedError
```

---

### `stratified_folds(y, n_folds) → folds`

Create stratified K-fold splits that preserve class proportions.

**Args:**
- `y`: Class labels, shape `(n_samples,)`
- `n_folds`: Number of folds

**Returns:**
- `folds`: List of tuples `(train_indices, val_indices)`

**Why Stratified?**
```
Original class distribution: 90% class 0, 10% class 1

Regular K-fold might create:
  Fold 1: 95% class 0, 5% class 1  (imbalanced!)
  Fold 2: 80% class 0, 20% class 1

Stratified K-fold ensures:
  All folds: ~90% class 0, ~10% class 1
```

**Implementation:**
```python
def stratified_folds(
    y: np.ndarray, n_folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified K-fold splits preserving class proportions.

    Args:
        y: Class labels, shape (n_samples,)
        n_folds: Number of folds

    Returns:
        folds: List of (train_indices, val_indices) tuples
    """
    # TODO:
    # classes = np.unique(y)
    # class_indices = {c: np.where(y == c)[0] for c in classes}
    #
    # # Create folds for each class separately
    # fold_indices = [[] for _ in range(n_folds)]
    # for c in classes:
    #     indices = class_indices[c]
    #     class_folds = create_folds(len(indices), n_folds)
    #     for fold_idx, (_, val_idx) in enumerate(class_folds):
    #         fold_indices[fold_idx].extend(indices[val_idx])
    #
    # # Convert to train/val splits
    # all_indices = np.arange(len(y))
    # folds = []
    # for val_idx in fold_indices:
    #     val_idx = np.array(val_idx)
    #     train_idx = np.setdiff1d(all_indices, val_idx)
    #     folds.append((train_idx, val_idx))
    # return folds
    raise NotImplementedError
```

---

### `cross_val_predict(model_fn, X, y, n_folds) → predictions`

Get cross-validated predictions for all samples.

**Args:**
- `model_fn`: Function that trains and returns a model
- `X`: Features, shape `(n_samples, n_features)`
- `y`: Targets, shape `(n_samples,)`
- `n_folds`: Number of folds

**Returns:**
- `predictions`: Predictions for all samples, shape `(n_samples,)`

**Note:** Each sample is predicted using a model trained without that sample.

**Implementation:**
```python
def cross_val_predict(
    model_fn: callable,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> np.ndarray:
    """
    Generate cross-validated predictions.

    Args:
        model_fn: Function that takes (X_train, y_train) and returns a predictor
        X: Features, shape (n_samples, n_features)
        y: Targets, shape (n_samples,)
        n_folds: Number of CV folds

    Returns:
        predictions: CV predictions, shape (n_samples,)
    """
    # TODO:
    # predictions = np.zeros(len(y))
    # folds = create_folds(len(X), n_folds)
    # for train_idx, val_idx in folds:
    #     model = model_fn(X[train_idx], y[train_idx])
    #     predictions[val_idx] = model(X[val_idx])
    # return predictions
    raise NotImplementedError
```

---

## Module 3: Model Selection

### `grid_search(model_fn, param_grid, X, y, n_folds, score_fn) → results`

Find best hyperparameters using grid search with cross-validation.

**Args:**
- `model_fn`: Function `(X, y, **params) → model` that trains with given params
- `param_grid`: Dictionary of parameter names to lists of values
- `X`: Features
- `y`: Targets
- `n_folds`: Number of CV folds
- `score_fn`: Scoring function

**Returns:**
- `results`: Dictionary with `best_params`, `best_score`, `all_results`

**Example:**
```python
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0],
    'l1_ratio': [0.0, 0.5, 1.0]
}

results = grid_search(train_elastic_net, param_grid, X, y)
# {
#   'best_params': {'alpha': 0.1, 'l1_ratio': 0.5},
#   'best_score': 0.87,
#   'all_results': [...]
# }
```

**Implementation:**
```python
def grid_search(
    model_fn: callable,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    score_fn: callable = None,
) -> dict:
    """
    Grid search with cross-validation.

    Args:
        model_fn: Function (X, y, **params) -> predictor
        param_grid: Dict of param_name -> list of values
        X: Features
        y: Targets
        n_folds: Number of CV folds
        score_fn: Scoring function

    Returns:
        results: Dict with best_params, best_score, all_results
    """
    # TODO:
    # from itertools import product
    #
    # param_names = list(param_grid.keys())
    # param_values = list(param_grid.values())
    #
    # best_score = -np.inf
    # best_params = None
    # all_results = []
    #
    # for values in product(*param_values):
    #     params = dict(zip(param_names, values))
    #
    #     def make_model(X_tr, y_tr, params=params):
    #         return model_fn(X_tr, y_tr, **params)
    #
    #     scores = cross_val_score(make_model, X, y, n_folds, score_fn)
    #     mean_score = np.mean(scores)
    #
    #     all_results.append({'params': params, 'mean_score': mean_score, 'scores': scores})
    #
    #     if mean_score > best_score:
    #         best_score = mean_score
    #         best_params = params
    #
    # return {'best_params': best_params, 'best_score': best_score, 'all_results': all_results}
    raise NotImplementedError
```

---

### `learning_curve(model_fn, X, y, train_sizes, n_folds, score_fn) → results`

Compute learning curve: scores vs training set size.

**Args:**
- `model_fn`: Model training function
- `X`: Features
- `y`: Targets
- `train_sizes`: List of training set sizes or fractions
- `n_folds`: Number of CV folds
- `score_fn`: Scoring function

**Returns:**
- `results`: Tuple `(train_sizes, train_scores, val_scores)`

**Use Case:**
```
High train score, low val score → Overfitting (need more data or regularization)
Both scores low → Underfitting (need more complex model)
Both scores high and close → Good fit
```

**Implementation:**
```python
def learning_curve(
    model_fn: callable,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: list[int | float],
    n_folds: int = 5,
    score_fn: callable = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute learning curve.

    Args:
        model_fn: Model training function
        X: Features
        y: Targets
        train_sizes: List of sizes (int) or fractions (float)
        n_folds: Number of CV folds
        score_fn: Scoring function

    Returns:
        train_sizes: Actual training sizes used
        train_scores: Mean training scores at each size
        val_scores: Mean validation scores at each size
    """
    # TODO:
    # n_samples = len(X)
    # actual_sizes = []
    # train_scores = []
    # val_scores = []
    #
    # for size in train_sizes:
    #     if isinstance(size, float):
    #         size = int(size * n_samples)
    #     actual_sizes.append(size)
    #
    #     # Use only first 'size' samples for this iteration
    #     X_subset, y_subset = X[:size], y[:size]
    #
    #     # CV on subset
    #     folds = create_folds(size, min(n_folds, size))
    #     fold_train_scores = []
    #     fold_val_scores = []
    #
    #     for train_idx, val_idx in folds:
    #         model = model_fn(X_subset[train_idx], y_subset[train_idx])
    #         fold_train_scores.append(score_fn(y_subset[train_idx], model(X_subset[train_idx])))
    #         fold_val_scores.append(score_fn(y_subset[val_idx], model(X_subset[val_idx])))
    #
    #     train_scores.append(np.mean(fold_train_scores))
    #     val_scores.append(np.mean(fold_val_scores))
    #
    # return np.array(actual_sizes), np.array(train_scores), np.array(val_scores)
    raise NotImplementedError
```

---

### `validation_curve(model_fn, X, y, param_name, param_range, n_folds, score_fn) → results`

Compute validation curve: scores vs hyperparameter value.

**Args:**
- `model_fn`: Function `(X, y, **{param_name: value}) → model`
- `X`: Features
- `y`: Targets
- `param_name`: Name of hyperparameter to vary
- `param_range`: List of values for the hyperparameter
- `n_folds`: Number of CV folds
- `score_fn`: Scoring function

**Returns:**
- `results`: Tuple `(param_range, train_scores, val_scores)`

**Use Case:**
```
Low param value: High train, low val → Overfitting
High param value: Both low → Underfitting
Optimal: Val score peaks
```

**Implementation:**
```python
def validation_curve(
    model_fn: callable,
    X: np.ndarray,
    y: np.ndarray,
    param_name: str,
    param_range: list,
    n_folds: int = 5,
    score_fn: callable = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute validation curve for a hyperparameter.

    Args:
        model_fn: Function (X, y, **params) -> predictor
        X: Features
        y: Targets
        param_name: Hyperparameter name
        param_range: Values to try
        n_folds: Number of CV folds
        score_fn: Scoring function

    Returns:
        param_range: Parameter values
        train_scores: Mean training scores
        val_scores: Mean validation scores
    """
    # TODO:
    # train_scores = []
    # val_scores = []
    #
    # for param_value in param_range:
    #     def make_model(X_tr, y_tr, pv=param_value):
    #         return model_fn(X_tr, y_tr, **{param_name: pv})
    #
    #     # Compute train and val scores via CV
    #     folds = create_folds(len(X), n_folds)
    #     fold_train = []
    #     fold_val = []
    #
    #     for train_idx, val_idx in folds:
    #         model = make_model(X[train_idx], y[train_idx])
    #         fold_train.append(score_fn(y[train_idx], model(X[train_idx])))
    #         fold_val.append(score_fn(y[val_idx], model(X[val_idx])))
    #
    #     train_scores.append(np.mean(fold_train))
    #     val_scores.append(np.mean(fold_val))
    #
    # return np.array(param_range), np.array(train_scores), np.array(val_scores)
    raise NotImplementedError
```

---

### `best_alpha_ridge(X, y, alphas, n_folds, score_fn) → best_alpha`

Find optimal Ridge regularization strength.

**Args:**
- `X`: Features
- `y`: Targets
- `alphas`: List of alpha values to try
- `n_folds`: Number of CV folds
- `score_fn`: Scoring function

**Returns:**
- `best_alpha`: Optimal regularization strength

**Convenience function:** Wraps grid_search for the common case of Ridge regression.

**Implementation:**
```python
def best_alpha_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alphas: list[float],
    n_folds: int = 5,
    score_fn: callable = None,
) -> float:
    """
    Find optimal Ridge alpha via cross-validation.

    Args:
        X: Features
        y: Targets
        alphas: List of alpha values to try
        n_folds: Number of CV folds
        score_fn: Scoring function

    Returns:
        best_alpha: Optimal regularization strength
    """
    # TODO:
    # best_score = -np.inf
    # best_alpha = alphas[0]
    #
    # for alpha in alphas:
    #     def train_ridge(X_tr, y_tr, a=alpha):
    #         # Train ridge regression and return predictor
    #         from stages.s06_linear_regression.starter.primitives import normal_equation
    #         # Add regularization to normal equation or use GD
    #         ...
    #
    #     scores = cross_val_score(train_ridge, X, y, n_folds, score_fn)
    #     mean_score = np.mean(scores)
    #
    #     if mean_score > best_score:
    #         best_score = mean_score
    #         best_alpha = alpha
    #
    # return best_alpha
    raise NotImplementedError
```

---

## Composition Example

```python
from stages.s09_regularization.starter.regularization import (
    ridge_penalty, ridge_gradient
)
from stages.s09_regularization.starter.cross_validation import (
    cross_val_score, create_folds
)
from stages.s09_regularization.starter.model_selection import grid_search

# Train Ridge regression with gradient descent
def train_ridge(X, y, alpha=1.0, lr=0.01, n_iter=1000):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(n_iter):
        y_pred = X @ w + b
        error = y_pred - y

        # MSE gradient + Ridge gradient
        grad_w = (2/len(y)) * X.T @ error + ridge_gradient(w, alpha)
        grad_b = (2/len(y)) * np.sum(error)

        w -= lr * grad_w
        b -= lr * grad_b

    return lambda X_new: X_new @ w + b

# Find best alpha
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
results = grid_search(train_ridge, param_grid, X, y, n_folds=5, score_fn=r2_score)

print(f"Best alpha: {results['best_params']['alpha']}")
print(f"Best CV score: {results['best_score']:.4f}")
```

---

## Constraints

### Allowed
- NumPy
- Python standard library (itertools)
- Previous stage implementations (s06 for regression primitives)

### Not Allowed
- Scikit-learn
- Any ML frameworks

### Code Style
- Type hints required
- Pure functions preferred
- Clear docstrings with examples

---

## Testing

```bash
pytest stages/s09_regularization/tests/ -v
python scripts/grade.py s09_regularization
```

---

## Success Criteria

- All penalty functions compute correct values
- Gradients match numerical differentiation
- Cross-validation produces consistent folds
- Grid search finds optimal hyperparameters
- Learning curves diagnose overfitting correctly
