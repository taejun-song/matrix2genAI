# Stage 8: Feature Engineering & Data Preprocessing - Specification

## Building Blocks to Implement

You will implement **12 simple functions** organized into three modules:

1. **Imputation** (3 functions) - Handle missing data
2. **Encoding** (4 functions) - Convert categorical data
3. **Scaling & Selection** (5 functions) - Transform and filter features

Each function is a pure building block with clear inputs and outputs.

---

## Module 1: Imputation

### `simple_imputer_strategy(X, strategy) → X_imputed`

Fill missing values (NaN) using specified strategy.

**Args:**
- `X`: Features with missing values, shape `(n_samples, n_features)`
- `strategy`: One of `'mean'`, `'median'`, or `'most_frequent'`

**Returns:**
- `X_imputed`: Features with NaN filled, shape `(n_samples, n_features)`

**Strategies:**
```
mean: Replace NaN with column mean (only for numeric data)
median: Replace NaN with column median (robust to outliers)
most_frequent: Replace NaN with mode (for any data type)
```

**Example:**
```python
X = np.array([[1.0, 2.0],
              [np.nan, 4.0],
              [3.0, np.nan]])

X_imputed = simple_imputer_strategy(X, strategy='mean')
# [[1.0, 2.0],
#  [2.0, 4.0],   # NaN replaced with mean([1.0, 3.0]) = 2.0
#  [3.0, 3.0]]   # NaN replaced with mean([2.0, 4.0]) = 3.0
```

**Implementation:**
```python
def simple_imputer_strategy(X: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Fill missing values using specified strategy.

    Args:
        X: Features with missing values, shape (n_samples, n_features)
        strategy: One of 'mean', 'median', 'most_frequent'

    Returns:
        X_imputed: Features with NaN filled, shape (n_samples, n_features)
    """
    # TODO: Implement imputation
    # 1. Copy X to avoid modifying original
    # 2. For each column:
    #    - Find NaN locations: np.isnan(X[:, col])
    #    - Compute statistic on non-NaN values
    #    - Fill NaN with computed value
    # 3. For 'most_frequent': use scipy.stats.mode or manual mode calculation

    raise NotImplementedError
```

---

### `find_missing_mask(X) → mask`

Create boolean mask indicating where data is missing.

**Args:**
- `X`: Features array, shape `(n_samples, n_features)`

**Returns:**
- `mask`: Boolean array, `True` where NaN, shape `(n_samples, n_features)`

**Example:**
```python
X = np.array([[1.0, np.nan],
              [2.0, 3.0],
              [np.nan, 4.0]])

mask = find_missing_mask(X)
# [[False, True],
#  [False, False],
#  [True, False]]
```

**Implementation:**
```python
def find_missing_mask(X: np.ndarray) -> np.ndarray:
    """
    Find where data is missing.

    Args:
        X: Features array, shape (n_samples, n_features)

    Returns:
        mask: Boolean mask, True where NaN, shape (n_samples, n_features)
    """
    # TODO: One line - use np.isnan()
    raise NotImplementedError
```

---

### `impute_with_constant(X, fill_value) → X_imputed`

Fill missing values with a constant.

**Args:**
- `X`: Features with missing values, shape `(n_samples, n_features)`
- `fill_value`: Scalar value to fill NaN with

**Returns:**
- `X_imputed`: Features with NaN filled, shape `(n_samples, n_features)`

**Example:**
```python
X = np.array([[1.0, np.nan],
              [np.nan, 3.0]])

X_imputed = impute_with_constant(X, fill_value=0.0)
# [[1.0, 0.0],
#  [0.0, 3.0]]
```

**Implementation:**
```python
def impute_with_constant(X: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Fill missing values with constant.

    Args:
        X: Features with missing values, shape (n_samples, n_features)
        fill_value: Value to fill NaN with

    Returns:
        X_imputed: Features with NaN filled, shape (n_samples, n_features)
    """
    # TODO: Use np.nan_to_num or manual replacement
    raise NotImplementedError
```

---

## Module 2: Encoding

### `label_encode(y) → (y_encoded, classes)`

Convert categorical labels to integers.

**Args:**
- `y`: Categorical labels, shape `(n_samples,)` - can be strings or any type

**Returns:**
- `y_encoded`: Integer labels, shape `(n_samples,)`, values in [0, n_classes-1]
- `classes`: Unique class labels in original form, shape `(n_classes,)`

**Example:**
```python
y = np.array(['cat', 'dog', 'cat', 'bird', 'dog'])

y_encoded, classes = label_encode(y)
# y_encoded = [0, 1, 0, 2, 1]
# classes = ['bird', 'cat', 'dog']  # Sorted alphabetically
```

**Implementation:**
```python
def label_encode(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode categorical labels as integers.

    Args:
        y: Categorical labels, shape (n_samples,)

    Returns:
        y_encoded: Integer labels, shape (n_samples,)
        classes: Unique classes, shape (n_classes,)
    """
    # TODO: Implement label encoding
    # 1. Find unique classes: np.unique(y)
    # 2. Create mapping from class to index
    # 3. Map each label to its index

    raise NotImplementedError
```

---

### `label_decode(y_encoded, classes) → y`

Convert integer labels back to original categories.

**Args:**
- `y_encoded`: Integer labels, shape `(n_samples,)`
- `classes`: Original class labels, shape `(n_classes,)`

**Returns:**
- `y`: Categorical labels, shape `(n_samples,)`

**Example:**
```python
y_encoded = np.array([0, 1, 0, 2])
classes = np.array(['bird', 'cat', 'dog'])

y = label_decode(y_encoded, classes)
# ['bird', 'cat', 'bird', 'dog']
```

**Implementation:**
```python
def label_decode(y_encoded: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    Decode integer labels back to categories.

    Args:
        y_encoded: Integer labels, shape (n_samples,)
        classes: Original class labels, shape (n_classes,)

    Returns:
        y: Categorical labels, shape (n_samples,)
    """
    # TODO: Use classes array as lookup table
    # return classes[y_encoded]
    raise NotImplementedError
```

---

### `one_hot_encode(y, n_classes) → y_onehot`

Convert integer labels to one-hot encoded matrix.

**Args:**
- `y`: Integer labels, shape `(n_samples,)`, values in [0, n_classes-1]
- `n_classes`: Number of classes (optional, auto-detect if None)

**Returns:**
- `y_onehot`: One-hot encoded matrix, shape `(n_samples, n_classes)`

**Example:**
```python
y = np.array([0, 1, 0, 2])

y_onehot = one_hot_encode(y, n_classes=3)
# [[1, 0, 0],
#  [0, 1, 0],
#  [1, 0, 0],
#  [0, 0, 1]]
```

**Implementation:**
```python
def one_hot_encode(y: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """
    Convert integer labels to one-hot encoding.

    Args:
        y: Integer labels, shape (n_samples,)
        n_classes: Number of classes (auto-detect if None)

    Returns:
        y_onehot: One-hot encoded, shape (n_samples, n_classes)
    """
    # TODO: Implement one-hot encoding
    # 1. If n_classes is None: n_classes = y.max() + 1
    # 2. Create zeros matrix: (n_samples, n_classes)
    # 3. Set appropriate elements to 1: y_onehot[range(n), y] = 1

    raise NotImplementedError
```

---

### `one_hot_decode(y_onehot) → y`

Convert one-hot encoded matrix back to integer labels.

**Args:**
- `y_onehot`: One-hot encoded matrix, shape `(n_samples, n_classes)`

**Returns:**
- `y`: Integer labels, shape `(n_samples,)`

**Example:**
```python
y_onehot = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])

y = one_hot_decode(y_onehot)
# [0, 1, 2]
```

**Implementation:**
```python
def one_hot_decode(y_onehot: np.ndarray) -> np.ndarray:
    """
    Convert one-hot encoding back to integer labels.

    Args:
        y_onehot: One-hot encoded, shape (n_samples, n_classes)

    Returns:
        y: Integer labels, shape (n_samples,)
    """
    # TODO: Find index of 1 in each row
    # return np.argmax(y_onehot, axis=1)
    raise NotImplementedError
```

---

## Module 3: Scaling & Selection

### `min_max_scale(X, feature_range) → (X_scaled, X_min, X_max)`

Scale features to specified range.

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `feature_range`: Tuple `(min, max)`, default `(0, 1)`

**Returns:**
- `X_scaled`: Scaled features, shape `(n_samples, n_features)`
- `X_min`: Minimum value per feature, shape `(n_features,)`
- `X_max`: Maximum value per feature, shape `(n_features,)`

**Formula:**
```
X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
```

**Example:**
```python
X = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])

X_scaled, X_min, X_max = min_max_scale(X, feature_range=(0, 1))
# X_scaled = [[0.0, 0.0],
#             [0.5, 0.5],
#             [1.0, 1.0]]
# X_min = [1.0, 2.0]
# X_max = [5.0, 6.0]
```

**Implementation:**
```python
def min_max_scale(
    X: np.ndarray, feature_range: tuple[float, float] = (0, 1)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale features to given range.

    Args:
        X: Features, shape (n_samples, n_features)
        feature_range: Target range (min, max)

    Returns:
        X_scaled: Scaled features, shape (n_samples, n_features)
        X_min: Min per feature, shape (n_features,)
        X_max: Max per feature, shape (n_features,)
    """
    # TODO: Implement min-max scaling
    # 1. Compute X_min, X_max per column
    # 2. Handle constant features (X_max == X_min)
    # 3. Apply formula above

    raise NotImplementedError
```

---

### `robust_scale(X) → (X_scaled, median, iqr)`

Scale features using median and IQR (robust to outliers).

**Args:**
- `X`: Features, shape `(n_samples, n_features)`

**Returns:**
- `X_scaled`: Scaled features, shape `(n_samples, n_features)`
- `median`: Median per feature, shape `(n_features,)`
- `iqr`: IQR per feature, shape `(n_features,)`

**Formula:**
```
X_scaled = (X - median) / IQR

where IQR = Q₃ - Q₁ (75th percentile - 25th percentile)
```

**Example:**
```python
X = np.array([[1.0], [2.0], [3.0], [100.0]])  # Has outlier

X_scaled, median, iqr = robust_scale(X)
# Outlier has less influence than with standardization
```

**Implementation:**
```python
def robust_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale features using median and IQR.

    Args:
        X: Features, shape (n_samples, n_features)

    Returns:
        X_scaled: Scaled features, shape (n_samples, n_features)
        median: Median per feature, shape (n_features,)
        iqr: IQR per feature, shape (n_features,)
    """
    # TODO: Implement robust scaling
    # 1. Compute median: np.median(X, axis=0)
    # 2. Compute Q1 and Q3: np.percentile(X, [25, 75], axis=0)
    # 3. IQR = Q3 - Q1
    # 4. Handle zero IQR (constant features)

    raise NotImplementedError
```

---

### `variance_threshold_select(X, threshold) → selected_features`

Select features with variance above threshold.

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `threshold`: Minimum variance, default `0.0`

**Returns:**
- `selected_features`: Boolean mask or indices, shape `(n_features,)`

**Formula:**
```
Var(X) = (1/n) Σᵢ (xᵢ - x̄)²

Keep feature j if Var(Xⱼ) > threshold
```

**Example:**
```python
X = np.array([[0, 1, 2],
              [0, 3, 4],
              [0, 5, 6]])  # First column has variance = 0

selected = variance_threshold_select(X, threshold=0.0)
# [False, True, True]  # Remove first column

X_selected = X[:, selected]
# [[1, 2],
#  [3, 4],
#  [5, 6]]
```

**Implementation:**
```python
def variance_threshold_select(
    X: np.ndarray, threshold: float = 0.0
) -> np.ndarray:
    """
    Select features with variance above threshold.

    Args:
        X: Features, shape (n_samples, n_features)
        threshold: Minimum variance

    Returns:
        selected_features: Boolean mask, shape (n_features,)
    """
    # TODO: Implement variance threshold
    # 1. Compute variance per column: np.var(X, axis=0)
    # 2. Return boolean mask: variances > threshold

    raise NotImplementedError
```

---

### `correlation_filter(X, threshold) → selected_features`

Remove features with correlation above threshold.

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `threshold`: Maximum absolute correlation, default `0.95`

**Returns:**
- `selected_features`: Boolean mask, shape `(n_features,)`

**Algorithm:**
```
1. Compute correlation matrix: ρᵢⱼ
2. For each pair (i, j) where i < j:
   - If |ρᵢⱼ| > threshold:
     - Remove feature j (keep i)
```

**Example:**
```python
X = np.array([[1, 2, 2.1],
              [2, 4, 4.2],
              [3, 6, 6.3]])  # Columns 1 and 2 are highly correlated

selected = correlation_filter(X, threshold=0.95)
# [True, True, False]  # Remove third column

X_selected = X[:, selected]
# [[1, 2],
#  [2, 4],
#  [3, 6]]
```

**Implementation:**
```python
def correlation_filter(X: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """
    Remove highly correlated features.

    Args:
        X: Features, shape (n_samples, n_features)
        threshold: Maximum absolute correlation

    Returns:
        selected_features: Boolean mask, shape (n_features,)
    """
    # TODO: Implement correlation filter
    # 1. Compute correlation matrix: np.corrcoef(X.T)
    # 2. Find pairs with |correlation| > threshold
    # 3. Keep first feature in each correlated pair

    raise NotImplementedError
```

---

### `detect_outliers_iqr(X, multiplier) → outlier_mask`

Detect outliers using IQR method.

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `multiplier`: IQR multiplier, default `1.5`

**Returns:**
- `outlier_mask`: Boolean mask, `True` for outliers, shape `(n_samples, n_features)`

**Formula:**
```
Q₁ = 25th percentile
Q₃ = 75th percentile
IQR = Q₃ - Q₁

Outlier if:
  x < Q₁ - multiplier × IQR  OR
  x > Q₃ + multiplier × IQR
```

**Example:**
```python
X = np.array([[1], [2], [3], [100]])  # 100 is outlier

outlier_mask = detect_outliers_iqr(X, multiplier=1.5)
# [[False],
#  [False],
#  [False],
#  [True]]
```

**Implementation:**
```python
def detect_outliers_iqr(
    X: np.ndarray, multiplier: float = 1.5
) -> np.ndarray:
    """
    Detect outliers using IQR method.

    Args:
        X: Features, shape (n_samples, n_features)
        multiplier: IQR multiplier (typically 1.5)

    Returns:
        outlier_mask: Boolean, True for outliers, shape (n_samples, n_features)
    """
    # TODO: Implement IQR outlier detection
    # 1. Compute Q1, Q3: np.percentile(X, [25, 75], axis=0)
    # 2. Compute IQR = Q3 - Q1
    # 3. Compute bounds
    # 4. Return mask of values outside bounds

    raise NotImplementedError
```

---

## Composition Example: Complete Pipeline

```python
# Load raw data with issues
X_train, y_train = load_messy_data()
X_test, y_test = load_messy_data(test=True)

# 1. Handle missing data (fit on train only)
X_train_clean = simple_imputer_strategy(X_train, strategy='median')
X_test_clean = simple_imputer_strategy(X_test, strategy='median')

# 2. Scale features (fit on train, apply to test)
X_train_scaled, train_min, train_max = min_max_scale(X_train_clean)
X_test_scaled = (X_test_clean - train_min) / (train_max - train_min)

# 3. Remove low-variance features (on train)
variance_mask = variance_threshold_select(X_train_scaled, threshold=0.01)
X_train_var = X_train_scaled[:, variance_mask]
X_test_var = X_test_scaled[:, variance_mask]

# 4. Remove correlated features (on train)
corr_mask = correlation_filter(X_train_var, threshold=0.9)
X_train_final = X_train_var[:, corr_mask]
X_test_final = X_test_var[:, corr_mask]

# 5. Encode labels
y_train_encoded, classes = label_encode(y_train)
# Use same classes for test
y_test_encoded = np.searchsorted(classes, y_test)

# Now ready for training!
model.fit(X_train_final, y_train_encoded)
```

---

## Constraints

### Allowed
- NumPy: All functions
- Python standard library
- For mode calculation: Can use `scipy.stats.mode` or implement manually

### Not Allowed
- Scikit-learn preprocessing (except for testing/comparison)
- Pandas
- Any preprocessing frameworks

### Code Style
- Pure functions (no side effects)
- Type hints for all functions
- Organize into 3 files:
  - `imputation.py`: Functions 1-3
  - `encoding.py`: Functions 4-7
  - `scaling.py`: Functions 8-12

---

## Testing

Your implementation will be tested on:

1. **Correctness**: Each function produces correct outputs
2. **Edge Cases**: Empty arrays, all NaN, constant features, single class
3. **Consistency**: Same results as scikit-learn (where applicable)
4. **Pipeline**: Functions compose correctly
5. **Data Leakage**: Test data not used for fitting

Run tests:
```bash
pytest stages/s08_feature_engineering/tests/ -v
python scripts/grade.py s08_feature_engineering
```

---

## Success Criteria

✅ All 12 functions pass unit tests
✅ Imputation handles all-NaN columns correctly
✅ Encoding invertible (encode → decode returns original)
✅ Scaling handles constant features without errors
✅ Feature selection removes appropriate features
✅ Pipeline example runs without errors
✅ No data leakage in test transformations

Good luck! Data preprocessing is the foundation of every ML project. 🚀
