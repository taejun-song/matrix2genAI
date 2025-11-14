# Stage 8: Feature Engineering & Data Preprocessing - Building Blocks

## Overview

Build a complete data preprocessing pipeline from simple, composable functions. Transform raw, messy data into clean features ready for machine learning.

**The Big Idea:** "Garbage in, garbage out" - even the best ML algorithm fails with bad features. Learn to clean, transform, and select features that make models work.

## Getting Started

### Setup

```bash
# Navigate to this stage
cd stages/s08_feature_engineering

# Run tests (using uv - recommended)
uv run pytest tests/ -v

# Or activate venv first
source .venv/bin/activate  # On Unix/macOS
pytest tests/ -v
```

### Files You'll Edit

- `starter/imputation.py` - Handle missing data
- `starter/encoding.py` - Convert categorical → numerical
- `starter/scaling.py` - Normalize, select features, detect outliers
- Tests in `tests/` verify your work

### Workflow

1. Read conceptual sections below
2. Implement sub-functions (1-3 lines, follow TODOs)
3. Test: `uv run pytest tests/test_imputation.py -v`
4. Compose into data pipelines!

---

## Learning Philosophy

You will implement **small building blocks** organized into 3 modules. Then you'll compose them into:
- Complete data cleaning pipelines
- Feature transformation workflows
- Dimensionality reduction systems

**Time:** 2-3 hours
**Difficulty:** ⭐⭐

## What You'll Build

### Core Building Blocks (12 functions across 3 modules)

**Module 1: Imputation (missing data handling)**
1. `simple_imputer_strategy(X, strategy)` - Fill missing values (mean/median/mode)
2. `find_missing_mask(X)` - Identify where data is missing
3. `impute_with_constant(X, fill_value)` - Fill with constant value

**Module 2: Encoding (categorical data)**
4. `label_encode(y)` - Convert categories to integers
5. `label_decode(y_encoded, classes)` - Convert back to categories
6. `one_hot_encode(y, n_classes)` - Create binary indicator matrix
7. `one_hot_decode(y_onehot)` - Convert one-hot back to labels

**Module 3: Scaling & Transformation**
8. `min_max_scale(X, feature_range)` - Scale to [min, max] range
9. `robust_scale(X)` - Scale using median and IQR (outlier-resistant)
10. `variance_threshold_select(X, threshold)` - Remove low-variance features
11. `correlation_filter(X, threshold)` - Remove highly correlated features
12. `detect_outliers_iqr(X, multiplier)` - Find outliers using IQR method

### What You'll Compose

Using these blocks, you'll write:
- End-to-end data cleaning pipelines
- Feature selection workflows
- Preprocessing for different data types

## Mathematical Background

### Missing Data

**Types of missingness:**
```
MCAR (Missing Completely At Random): P(missing | data) = P(missing)
MAR (Missing At Random): P(missing | data) = P(missing | observed)
MNAR (Missing Not At Random): Missingness depends on unobserved values
```

**Imputation strategies:**
```
Mean: x̄ = (1/n) Σᵢ xᵢ  (for observed values only)
Median: Middle value when sorted
Mode: Most frequent value
```

### One-Hot Encoding

```
Input: y = ["cat", "dog", "cat", "bird"]
Label encoding: [0, 1, 0, 2]

One-hot encoding:
    cat  dog  bird
  [[1,   0,   0],
   [0,   1,   0],
   [1,   0,   0],
   [0,   0,   1]]
```

### Min-Max Scaling

```
X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min

For range [0, 1]:
X_scaled = (X - X_min) / (X_max - X_min)
```

### Robust Scaling

```
X_scaled = (X - median(X)) / IQR(X)

where IQR = Q₃ - Q₁ (75th percentile - 25th percentile)
```

### Variance Threshold

```
Remove features where:
Var(X) = (1/n) Σᵢ (xᵢ - x̄)² < threshold

Low variance → feature doesn't vary much → likely not informative
```

### Correlation Filter

```
Correlation between features i and j:
ρᵢⱼ = Cov(Xᵢ, Xⱼ) / (σᵢ σⱼ)

If |ρᵢⱼ| > threshold → features are redundant → remove one
```

### Outlier Detection (IQR Method)

```
Q₁ = 25th percentile
Q₃ = 75th percentile
IQR = Q₃ - Q₁

Outliers are values where:
x < Q₁ - multiplier × IQR  OR  x > Q₃ + multiplier × IQR

(typical multiplier = 1.5)
```

## Implementation Guide

### Step 1: Missing Data Imputation (30 min)

**Example: `simple_imputer_strategy`**
```python
def simple_imputer_strategy(X, strategy='mean'):
    X_copy = X.copy()
    mask = np.isnan(X_copy)

    for col in range(X_copy.shape[1]):
        col_mask = mask[:, col]
        if not col_mask.any():
            continue

        if strategy == 'mean':
            fill_value = np.nanmean(X_copy[:, col])
        elif strategy == 'median':
            fill_value = np.nanmedian(X_copy[:, col])
        elif strategy == 'most_frequent':
            # Mode implementation
            pass

        X_copy[col_mask, col] = fill_value

    return X_copy
```

### Step 2: Categorical Encoding (30 min)

**Example: `one_hot_encode`**
```python
def one_hot_encode(y, n_classes=None):
    if n_classes is None:
        n_classes = len(np.unique(y))

    n_samples = len(y)
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y] = 1

    return y_onehot
```

### Step 3: Scaling & Selection (45 min)

**Example: `min_max_scale`**
```python
def min_max_scale(X, feature_range=(0, 1)):
    min_val, max_val = feature_range
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    # Handle constant features
    range_mask = (X_max - X_min) != 0

    X_scaled = X.copy()
    X_scaled[:, range_mask] = (X[:, range_mask] - X_min[range_mask]) / \
                               (X_max[range_mask] - X_min[range_mask])
    X_scaled = X_scaled * (max_val - min_val) + min_val

    return X_scaled, X_min, X_max
```

### Step 4: Compose Pipeline (30 min)

```python
# Load messy data
X_train, y_train = load_messy_data()  # Has missing values, mixed scales

# 1. Handle missing data
X_train = simple_imputer_strategy(X_train, strategy='median')

# 2. Detect and handle outliers
outlier_mask = detect_outliers_iqr(X_train, multiplier=1.5)
# Option: remove outliers or cap them

# 3. Scale features
X_train_scaled, min_vals, max_vals = min_max_scale(X_train)

# 4. Remove low-variance features
selected_features = variance_threshold_select(X_train_scaled, threshold=0.01)
X_train_selected = X_train_scaled[:, selected_features]

# 5. Remove correlated features
final_features = correlation_filter(X_train_selected, threshold=0.95)
X_train_final = X_train_selected[:, final_features]

# Apply same transforms to test data
X_test = simple_imputer_strategy(X_test, strategy='median')
X_test = (X_test - min_vals) / (max_vals - min_vals)  # Same scaling
X_test = X_test[:, selected_features][:, final_features]  # Same features
```

## Key Concepts

### Why Imputation Matters

```
Original data:
Age: [25, 30, NaN, 35, 40]

Without imputation → Can't use this sample!

With mean imputation:
Age: [25, 30, 32.5, 35, 40]  # Now we can train on all samples
```

### When to Use Each Scaling Method

**Min-Max Scaling:**
- ✅ Neural networks (gradients work better in [0,1] or [-1,1])
- ✅ Distance-based algorithms (KNN, K-means)
- ❌ When you have outliers (they compress the rest of the data)

**Standardization (from s06):**
- ✅ Linear regression, logistic regression
- ✅ PCA, LDA
- ❌ When features have different meaningful ranges

**Robust Scaling:**
- ✅ When you have outliers
- ✅ When median is more representative than mean
- ✅ Financial data, sensor data

### One-Hot vs Label Encoding

**Label Encoding:** `["red", "blue", "green"]` → `[0, 1, 2]`
- ✅ For ordinal categories: ["low", "medium", "high"]
- ❌ For nominal categories: Creates artificial ordering

**One-Hot Encoding:** `["red", "blue", "green"]` → `[[1,0,0], [0,1,0], [0,0,1]]`
- ✅ For nominal categories: No artificial ordering
- ❌ High dimensionality with many categories

### Feature Selection vs Feature Extraction

**Feature Selection:** Choose subset of existing features
- Variance threshold: Remove low-variance features
- Correlation filter: Remove redundant features
- Maintains interpretability

**Feature Extraction:** Create new features
- PCA: Linear combinations of features
- May lose interpretability but capture more information

## Common Pitfalls

### 1. Fitting on Test Data

```python
# ❌ Bad: Different scaling for train and test
X_train_scaled, _, _ = min_max_scale(X_train)
X_test_scaled, _, _ = min_max_scale(X_test)  # Different min/max!

# ✅ Good: Use training statistics for test
X_train_scaled, min_vals, max_vals = min_max_scale(X_train)
X_test_scaled = (X_test - min_vals) / (max_vals - min_vals)
```

### 2. Leaking Information

```python
# ❌ Bad: Imputing before split
X = simple_imputer_strategy(X_all, strategy='mean')
X_train, X_test = train_test_split(X)  # Test mean leaked into train!

# ✅ Good: Impute after split
X_train, X_test = train_test_split(X_all)
X_train = simple_imputer_strategy(X_train, strategy='mean')
X_test = simple_imputer_strategy(X_test, strategy='mean')  # Or use train stats
```

### 3. Not Handling Constant Features

```python
# ❌ Bad: Division by zero
X_scaled = (X - X.min()) / (X.max() - X.min())  # Fails if max == min!

# ✅ Good: Check for constant features
range_vals = X.max() - X.min()
if (range_vals == 0).any():
    # Handle or remove constant features
```

### 4. Wrong Imputation for Data Type

```python
# ❌ Bad: Mean for categorical
categories = ["A", "B", "A", NaN, "C"]
# Can't compute mean of strings!

# ✅ Good: Mode for categorical
# Most frequent value: "A"
```

## Experiments to Try

### 1. Compare Imputation Strategies

```python
for strategy in ['mean', 'median', 'most_frequent']:
    X_imputed = simple_imputer_strategy(X_with_missing, strategy)
    # Train model and compare performance
    # Which works best for your data?
```

### 2. Outlier Impact

```python
# With outliers
X_with_outliers = X.copy()
acc_before = train_and_evaluate(X_with_outliers, y)

# Without outliers
outliers = detect_outliers_iqr(X, multiplier=1.5)
X_clean = X[~outliers.any(axis=1)]
y_clean = y[~outliers.any(axis=1)]
acc_after = train_and_evaluate(X_clean, y_clean)

print(f"Accuracy improved by: {acc_after - acc_before:.3f}")
```

### 3. Feature Selection Impact

```python
# All features
acc_all = train_and_evaluate(X, y)

# After variance threshold
selected = variance_threshold_select(X, threshold=0.01)
acc_variance = train_and_evaluate(X[:, selected], y)

# After correlation filter
final = correlation_filter(X[:, selected], threshold=0.9)
acc_final = train_and_evaluate(X[:, selected][:, final], y)

# Did we lose performance? Gain speed?
```

## Debugging Guide

### Missing values not handled

```python
# Check for NaN
print(f"Has NaN: {np.isnan(X).any()}")
print(f"NaN count per column: {np.isnan(X).sum(axis=0)}")

# Check for inf
print(f"Has inf: {np.isinf(X).any()}")

# Visualize missing pattern
import matplotlib.pyplot as plt
plt.imshow(np.isnan(X), aspect='auto', cmap='gray')
plt.xlabel('Features')
plt.ylabel('Samples')
plt.title('Missing Data Pattern (white = missing)')
```

### Scaling produces strange values

```python
# Check for constant features
print(f"Constant features: {(X.std(axis=0) == 0).sum()}")

# Check range
print(f"Min: {X.min(axis=0)}")
print(f"Max: {X.max(axis=0)}")

# Check for outliers
Q1 = np.percentile(X, 25, axis=0)
Q3 = np.percentile(X, 75, axis=0)
IQR = Q3 - Q1
outliers = ((X < Q1 - 1.5*IQR) | (X > Q3 + 1.5*IQR)).sum(axis=0)
print(f"Outliers per feature: {outliers}")
```

### One-hot encoding dimension explosion

```python
# Check unique values
n_unique = len(np.unique(y))
print(f"Number of categories: {n_unique}")

if n_unique > 50:
    print("Warning: Too many categories for one-hot encoding!")
    print("Consider: label encoding, target encoding, or feature hashing")
```

## Testing Your Implementation

```bash
# Test individual functions
pytest stages/s08_feature_engineering/tests/test_imputation.py -v
pytest stages/s08_feature_engineering/tests/test_encoding.py -v
pytest stages/s08_feature_engineering/tests/test_scaling.py -v

# Test full pipeline
pytest stages/s08_feature_engineering/tests/test_pipeline.py -v

# Grade your work
python scripts/grade.py s08_feature_engineering
```

## Real-World Applications

Feature engineering is critical in industry:

- **Kaggle Competitions**: Winners spend 80% of time on feature engineering
- **Production ML**: Data quality determines model quality
- **Time Series**: Handle missing sensor readings
- **E-commerce**: Encode categorical product features
- **Healthcare**: Impute missing patient data
- **Finance**: Detect and handle outlier transactions

**Common data quality issues:**
- 20-30% of data has missing values
- Categorical variables need encoding
- Features often have vastly different scales
- Outliers from measurement errors

## What's Next

After mastering feature engineering:

**s09: Regularization** - Prevent overfitting with L1/L2
**s11: Neural Networks** - Feature learning (automated feature engineering!)
**s24: Autoencoders** - Learn compressed representations

The pattern stays the same:
1. Simple building block functions
2. Test each independently
3. Compose into pipelines

## Success Criteria

You understand this stage when you can:

- ✅ Explain when to use mean vs median imputation
- ✅ Choose appropriate encoding for different data types
- ✅ Apply transformations consistently to train and test data
- ✅ Detect and handle outliers appropriately
- ✅ Build complete preprocessing pipelines
- ✅ Debug data quality issues

**Target: 90%+ test passing rate**

Remember: "Garbage in, garbage out" - good preprocessing is essential! 🚀
