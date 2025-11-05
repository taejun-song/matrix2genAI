# Stage 7: Logistic Regression & Classification - Specification

## Building Blocks to Implement

You will implement **15 simple functions** organized into three modules:

1. **Activations & Predictions** (6 functions)
2. **Loss & Gradients** (4 functions)
3. **Metrics** (5 functions)

Each function is a pure building block with clear inputs and outputs.

---

## Module 1: Activations & Predictions

### `sigmoid(z) → probabilities`

Apply sigmoid activation function.

**Args:**
- `z`: Linear outputs, shape `(n_samples,)` or `(n_samples, n_classes)`

**Returns:**
- `probabilities`: Sigmoid outputs, shape matches input, range (0, 1)

**Formula:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Numerical Stability:**
- Clip `z` to `[-500, 500]` to avoid overflow

**Example:**
```python
z = np.array([-1, 0, 1, 10])
probs = sigmoid(z)
# [0.269, 0.5, 0.731, 0.9999...]
```

**Implementation:**
```python
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Apply sigmoid activation function.

    Args:
        z: Linear outputs, any shape

    Returns:
        probabilities: Sigmoid applied element-wise, same shape as z
    """
    # TODO: Implement sigmoid with numerical stability
    # 1. Clip z to avoid overflow: np.clip(z, -500, 500)
    # 2. Return 1 / (1 + np.exp(-z))

    pass
```

---

### `softmax(z) → probabilities`

Apply softmax activation for multi-class classification.

**Args:**
- `z`: Linear outputs, shape `(n_samples, n_classes)`

**Returns:**
- `probabilities`: Softmax outputs, shape `(n_samples, n_classes)`
  - Each row sums to 1
  - Each element in (0, 1)

**Formula:**
```
softmax(z)ⱼ = e^(zⱼ) / Σₖ e^(zₖ)
```

**Numerical Stability:**
- Subtract max from each row before exp: `z - max(z, axis=-1, keepdims=True)`

**Example:**
```python
z = np.array([[1, 2, 3], [1, 1, 1]])
probs = softmax(z)
# [[0.09, 0.24, 0.67],   # Sum = 1.0
#  [0.33, 0.33, 0.33]]   # Sum = 1.0
```

**Implementation:**
```python
def softmax(z: np.ndarray) -> np.ndarray:
    """
    Apply softmax activation function.

    Args:
        z: Linear outputs, shape (n_samples, n_classes)

    Returns:
        probabilities: Softmax outputs, shape (n_samples, n_classes)
    """
    # TODO: Implement softmax with numerical stability
    # 1. Subtract max: z_stable = z - np.max(z, axis=-1, keepdims=True)
    # 2. Compute exp: exp_z = np.exp(z_stable)
    # 3. Normalize: exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    pass
```

---

### `predict_proba_binary(X, weights, bias) → probabilities`

Compute binary class probabilities.

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `weights`: Weight vector, shape `(n_features,)`
- `bias`: Bias term (scalar)

**Returns:**
- `probabilities`: P(y=1|X), shape `(n_samples,)`

**Formula:**
```
z = Xw + b
P(y=1|X) = σ(z)
```

**Example:**
```python
X = np.array([[1, 2], [3, 4]])
weights = np.array([0.5, 0.5])
bias = 0.0

probs = predict_proba_binary(X, weights, bias)
# [0.88, 0.99]  # High confidence for class 1
```

**Implementation:**
```python
def predict_proba_binary(
    X: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    """
    Compute binary class probabilities.

    Args:
        X: Features, shape (n_samples, n_features)
        weights: Weight vector, shape (n_features,)
        bias: Bias term (scalar)

    Returns:
        probabilities: P(y=1|X), shape (n_samples,)
    """
    # TODO: Compose linear prediction + sigmoid
    # z = X @ weights + bias
    # return sigmoid(z)

    pass
```

---

### `predict_proba_multiclass(X, weights, bias) → probabilities`

Compute multi-class probabilities.

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `weights`: Weight matrix, shape `(n_features, n_classes)`
- `bias`: Bias vector, shape `(n_classes,)`

**Returns:**
- `probabilities`: Class probabilities, shape `(n_samples, n_classes)`

**Formula:**
```
Z = XW + b
P(y=j|X) = softmax(Z)
```

**Example:**
```python
X = np.array([[1, 2], [3, 4]])
weights = np.array([[0.5, 0.3], [0.2, 0.4]])  # 2 features, 2 classes
bias = np.array([0.0, 0.1])

probs = predict_proba_multiclass(X, weights, bias)
# [[0.43, 0.57],  # Class 0: 43%, Class 1: 57%
#  [0.35, 0.65]]  # Class 0: 35%, Class 1: 65%
```

**Implementation:**
```python
def predict_proba_multiclass(
    X: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """
    Compute multi-class probabilities.

    Args:
        X: Features, shape (n_samples, n_features)
        weights: Weight matrix, shape (n_features, n_classes)
        bias: Bias vector, shape (n_classes,)

    Returns:
        probabilities: Class probabilities, shape (n_samples, n_classes)
    """
    # TODO: Compose linear prediction + softmax
    # Z = X @ weights + bias
    # return softmax(Z)

    pass
```

---

### `predict_binary(X, weights, bias, threshold) → predictions`

Make binary predictions.

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `weights`: Weight vector, shape `(n_features,)`
- `bias`: Bias term (scalar)
- `threshold`: Decision threshold, default 0.5

**Returns:**
- `predictions`: Binary labels {0, 1}, shape `(n_samples,)`

**Formula:**
```
P(y=1|X) = predict_proba_binary(X, weights, bias)
ŷ = 1 if P(y=1|X) ≥ threshold else 0
```

**Example:**
```python
X = np.array([[1, 2], [3, 4]])
weights = np.array([0.5, 0.5])
bias = 0.0

preds = predict_binary(X, weights, bias, threshold=0.5)
# [1, 1]  # Both above threshold
```

**Implementation:**
```python
def predict_binary(
    X: np.ndarray, weights: np.ndarray, bias: float, threshold: float = 0.5
) -> np.ndarray:
    """
    Make binary predictions.

    Args:
        X: Features, shape (n_samples, n_features)
        weights: Weight vector, shape (n_features,)
        bias: Bias term (scalar)
        threshold: Decision threshold

    Returns:
        predictions: Binary labels {0, 1}, shape (n_samples,)
    """
    # TODO: Get probabilities and threshold
    # probs = predict_proba_binary(X, weights, bias)
    # return (probs >= threshold).astype(int)

    pass
```

---

### `predict_multiclass(X, weights, bias) → predictions`

Make multi-class predictions.

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `weights`: Weight matrix, shape `(n_features, n_classes)`
- `bias`: Bias vector, shape `(n_classes,)`

**Returns:**
- `predictions`: Class indices, shape `(n_samples,)`

**Formula:**
```
P = predict_proba_multiclass(X, weights, bias)
ŷ = argmax(P, axis=1)
```

**Example:**
```python
X = np.array([[1, 2], [3, 4]])
weights = np.array([[0.5, 0.3, 0.1], [0.2, 0.4, 0.5]])  # 3 classes
bias = np.array([0.0, 0.0, 0.0])

preds = predict_multiclass(X, weights, bias)
# [1, 2]  # Class 1 for first sample, class 2 for second
```

**Implementation:**
```python
def predict_multiclass(
    X: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """
    Make multi-class predictions.

    Args:
        X: Features, shape (n_samples, n_features)
        weights: Weight matrix, shape (n_features, n_classes)
        bias: Bias vector, shape (n_classes,)

    Returns:
        predictions: Class indices, shape (n_samples,)
    """
    # TODO: Get probabilities and take argmax
    # probs = predict_proba_multiclass(X, weights, bias)
    # return np.argmax(probs, axis=1)

    pass
```

---

## Module 2: Loss & Gradients

### `binary_cross_entropy(y_true, y_pred_proba) → loss`

Compute binary cross-entropy loss.

**Args:**
- `y_true`: True labels {0, 1}, shape `(n_samples,)`
- `y_pred_proba`: Predicted probabilities P(y=1), shape `(n_samples,)`

**Returns:**
- `loss`: Scalar value

**Formula:**
```
L = -(1/n) Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
```

**Numerical Stability:**
- Clip probabilities to `[1e-15, 1 - 1e-15]` to avoid log(0)

**Example:**
```python
y_true = np.array([1, 0, 1])
y_pred_proba = np.array([0.9, 0.1, 0.8])

loss = binary_cross_entropy(y_true, y_pred_proba)
# ~0.11  # Low loss, good predictions
```

**Implementation:**
```python
def binary_cross_entropy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute binary cross-entropy loss.

    Args:
        y_true: True labels {0, 1}, shape (n_samples,)
        y_pred_proba: Predicted probabilities, shape (n_samples,)

    Returns:
        loss: Binary cross-entropy loss (scalar)
    """
    # TODO: Implement binary cross-entropy
    # 1. Clip probabilities: eps = 1e-15; p = np.clip(y_pred_proba, eps, 1-eps)
    # 2. Compute loss: -np.mean(y_true * np.log(p) + (1-y_true) * np.log(1-p))

    pass
```

---

### `categorical_cross_entropy(y_true, y_pred_proba) → loss`

Compute categorical cross-entropy loss.

**Args:**
- `y_true`: True labels (one-hot), shape `(n_samples, n_classes)`
- `y_pred_proba`: Predicted probabilities, shape `(n_samples, n_classes)`

**Returns:**
- `loss`: Scalar value

**Formula:**
```
L = -(1/n) Σᵢ Σⱼ yᵢⱼ log(pᵢⱼ)
```

**Numerical Stability:**
- Clip probabilities to `[1e-15, 1]` to avoid log(0)

**Example:**
```python
y_true = np.array([[1, 0, 0], [0, 1, 0]])  # One-hot encoded
y_pred_proba = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])

loss = categorical_cross_entropy(y_true, y_pred_proba)
# ~0.12  # Low loss, good predictions
```

**Implementation:**
```python
def categorical_cross_entropy(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> float:
    """
    Compute categorical cross-entropy loss.

    Args:
        y_true: True labels (one-hot), shape (n_samples, n_classes)
        y_pred_proba: Predicted probabilities, shape (n_samples, n_classes)

    Returns:
        loss: Categorical cross-entropy loss (scalar)
    """
    # TODO: Implement categorical cross-entropy
    # 1. Clip probabilities: eps = 1e-15; p = np.clip(y_pred_proba, eps, 1)
    # 2. Compute loss: -np.mean(np.sum(y_true * np.log(p), axis=1))

    pass
```

---

### `binary_cross_entropy_gradient(X, y_true, y_pred_proba) → (grad_w, grad_b)`

Compute gradients of binary cross-entropy loss.

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `y_true`: True labels {0, 1}, shape `(n_samples,)`
- `y_pred_proba`: Predicted probabilities, shape `(n_samples,)`

**Returns:**
- `grad_w`: Gradient w.r.t. weights, shape `(n_features,)`
- `grad_b`: Gradient w.r.t. bias (scalar)

**Formulas:**
```
∂L/∂w = (1/n) Xᵀ(p - y)
∂L/∂b = (1/n) Σᵢ(pᵢ - yᵢ)

where p = σ(Xw + b)
```

**Derivation:**
The sigmoid derivative cancels nicely with the cross-entropy derivative!

**Example:**
```python
X = np.array([[1, 2], [3, 4]])
y_true = np.array([1, 0])
y_pred_proba = np.array([0.9, 0.2])

grad_w, grad_b = binary_cross_entropy_gradient(X, y_true, y_pred_proba)
# grad_w ≈ [0.35, 0.5]  # Direction to improve weights
# grad_b ≈ 0.05         # Direction to improve bias
```

**Implementation:**
```python
def binary_cross_entropy_gradient(
    X: np.ndarray, y_true: np.ndarray, y_pred_proba: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Compute gradients of binary cross-entropy loss.

    Args:
        X: Features, shape (n_samples, n_features)
        y_true: True labels {0, 1}, shape (n_samples,)
        y_pred_proba: Predicted probabilities, shape (n_samples,)

    Returns:
        grad_w: Gradient w.r.t. weights, shape (n_features,)
        grad_b: Gradient w.r.t. bias (scalar)
    """
    n = len(y_true)
    errors = y_pred_proba - y_true

    # TODO: Compute gradients
    # grad_w = (1/n) * X.T @ errors
    # grad_b = (1/n) * np.sum(errors)

    pass
```

---

### `categorical_cross_entropy_gradient(X, y_true, y_pred_proba) → (grad_w, grad_b)`

Compute gradients of categorical cross-entropy loss.

**Args:**
- `X`: Features, shape `(n_samples, n_features)`
- `y_true`: True labels (one-hot), shape `(n_samples, n_classes)`
- `y_pred_proba`: Predicted probabilities, shape `(n_samples, n_classes)`

**Returns:**
- `grad_w`: Gradient w.r.t. weights, shape `(n_features, n_classes)`
- `grad_b`: Gradient w.r.t. bias, shape `(n_classes,)`

**Formulas:**
```
∂L/∂W = (1/n) Xᵀ(P - Y)
∂L/∂b = (1/n) Σᵢ(pᵢ - yᵢ)

where P = softmax(XW + b), Y is one-hot
```

**Example:**
```python
X = np.array([[1, 2], [3, 4]])
y_true = np.array([[1, 0], [0, 1]])  # One-hot
y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8]])

grad_w, grad_b = categorical_cross_entropy_gradient(X, y_true, y_pred_proba)
# grad_w: shape (2, 2)
# grad_b: shape (2,)
```

**Implementation:**
```python
def categorical_cross_entropy_gradient(
    X: np.ndarray, y_true: np.ndarray, y_pred_proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute gradients of categorical cross-entropy loss.

    Args:
        X: Features, shape (n_samples, n_features)
        y_true: True labels (one-hot), shape (n_samples, n_classes)
        y_pred_proba: Predicted probabilities, shape (n_samples, n_classes)

    Returns:
        grad_w: Gradient w.r.t. weights, shape (n_features, n_classes)
        grad_b: Gradient w.r.t. bias, shape (n_classes,)
    """
    n = len(y_true)
    errors = y_pred_proba - y_true

    # TODO: Compute gradients
    # grad_w = (1/n) * X.T @ errors
    # grad_b = (1/n) * np.sum(errors, axis=0)

    pass
```

---

## Module 3: Metrics

### `accuracy(y_true, y_pred) → accuracy`

Compute classification accuracy.

**Args:**
- `y_true`: True labels, shape `(n_samples,)`
- `y_pred`: Predicted labels, shape `(n_samples,)`

**Returns:**
- `accuracy`: Fraction correct, range [0, 1]

**Formula:**
```
Accuracy = (1/n) Σᵢ 𝟙(ŷᵢ = yᵢ)
```

**Example:**
```python
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0])

acc = accuracy(y_true, y_pred)
# 0.8  # 4/5 correct
```

**Implementation:**
```python
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)

    Returns:
        accuracy: Fraction of correct predictions
    """
    # TODO: One line
    # return np.mean(y_true == y_pred)

    pass
```

---

### `confusion_matrix(y_true, y_pred, n_classes) → matrix`

Compute confusion matrix.

**Args:**
- `y_true`: True labels, shape `(n_samples,)`
- `y_pred`: Predicted labels, shape `(n_samples,)`
- `n_classes`: Number of classes (optional, auto-detect if None)

**Returns:**
- `matrix`: Confusion matrix, shape `(n_classes, n_classes)`
  - `matrix[i, j]` = count of true class i predicted as class j

**Example (Binary):**
```python
y_true = np.array([1, 0, 1, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1])

cm = confusion_matrix(y_true, y_pred, n_classes=2)
# [[2, 1],   TN=2, FP=1
#  [1, 2]]   FN=1, TP=2
```

**Example (Multi-class):**
```python
y_true = np.array([0, 1, 2, 1, 0])
y_pred = np.array([0, 1, 1, 1, 0])

cm = confusion_matrix(y_true, y_pred, n_classes=3)
# [[2, 0, 0],
#  [0, 2, 0],
#  [0, 1, 0]]
```

**Implementation:**
```python
def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        n_classes: Number of classes (auto-detect if None)

    Returns:
        matrix: Confusion matrix, shape (n_classes, n_classes)
    """
    # TODO: Build confusion matrix
    # if n_classes is None:
    #     n_classes = max(y_true.max(), y_pred.max()) + 1
    #
    # cm = np.zeros((n_classes, n_classes), dtype=int)
    # for true_label, pred_label in zip(y_true, y_pred):
    #     cm[true_label, pred_label] += 1
    # return cm

    pass
```

---

### `precision_recall_f1(y_true, y_pred, average) → (precision, recall, f1)`

Compute precision, recall, and F1 score.

**Args:**
- `y_true`: True labels, shape `(n_samples,)`
- `y_pred`: Predicted labels, shape `(n_samples,)`
- `average`: 'binary' or 'macro' (for multi-class)

**Returns:**
- `precision`: Precision score
- `recall`: Recall score
- `f1`: F1 score

**Formulas (Binary):**
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * (Precision * Recall) / (Precision + Recall)
```

**For Multi-class (macro):**
- Compute per-class precision/recall
- Average across classes

**Edge Cases:**
- If TP + FP = 0, set precision = 0
- If TP + FN = 0, set recall = 0
- If precision + recall = 0, set F1 = 0

**Example:**
```python
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0])

precision, recall, f1 = precision_recall_f1(y_true, y_pred, average='binary')
# precision = 1.0  # 2 TP, 0 FP
# recall = 0.67    # 2 TP, 1 FN
# f1 = 0.8
```

**Implementation:**
```python
def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary'
) -> tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        average: 'binary' or 'macro'

    Returns:
        precision: Precision score
        recall: Recall score
        f1: F1 score
    """
    # TODO: Implement precision/recall/F1
    # For binary:
    #   TP = np.sum((y_true == 1) & (y_pred == 1))
    #   FP = np.sum((y_true == 0) & (y_pred == 1))
    #   FN = np.sum((y_true == 1) & (y_pred == 0))
    #   precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    #   recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    #   f1 = 2*P*R/(P+R) if (P+R) > 0 else 0

    pass
```

---

### `roc_auc_score(y_true, y_pred_proba) → auc`

Compute ROC AUC score (binary classification only).

**Args:**
- `y_true`: True binary labels {0, 1}, shape `(n_samples,)`
- `y_pred_proba`: Predicted probabilities for class 1, shape `(n_samples,)`

**Returns:**
- `auc`: Area under ROC curve, range [0, 1]

**Algorithm:**
1. Sort samples by predicted probability (descending)
2. Compute TPR and FPR at each threshold
3. Use trapezoidal rule to compute area under curve

**Example:**
```python
y_true = np.array([1, 0, 1, 0])
y_pred_proba = np.array([0.9, 0.3, 0.8, 0.2])

auc = roc_auc_score(y_true, y_pred_proba)
# 1.0  # Perfect ranking
```

**Implementation:**
```python
def roc_auc_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute ROC AUC score.

    Args:
        y_true: True binary labels {0, 1}, shape (n_samples,)
        y_pred_proba: Predicted probabilities, shape (n_samples,)

    Returns:
        auc: Area under ROC curve
    """
    # TODO: Implement ROC AUC
    # Hint: Use sklearn.metrics.roc_auc_score for simplicity,
    # or implement from scratch:
    # 1. Sort by predicted probability
    # 2. Compute TPR/FPR at each threshold
    # 3. Use trapezoidal rule (np.trapz) for area

    pass
```

---

### `classification_report(y_true, y_pred, y_pred_proba) → dict`

Generate complete classification report.

**Args:**
- `y_true`: True labels, shape `(n_samples,)`
- `y_pred`: Predicted labels, shape `(n_samples,)`
- `y_pred_proba`: Predicted probabilities (optional for binary)

**Returns:**
- `report`: Dictionary with all metrics

**Example:**
```python
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0])
y_pred_proba = np.array([0.9, 0.1, 0.4, 0.8, 0.2])

report = classification_report(y_true, y_pred, y_pred_proba)
# {
#     'accuracy': 0.8,
#     'precision': 1.0,
#     'recall': 0.67,
#     'f1': 0.8,
#     'roc_auc': 0.89,
#     'confusion_matrix': [[2, 0], [1, 2]]
# }
```

**Implementation:**
```python
def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None
) -> dict:
    """
    Generate complete classification report.

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        y_pred_proba: Predicted probabilities (optional)

    Returns:
        report: Dictionary with all metrics
    """
    # TODO: Compose all metrics
    # report = {
    #     'accuracy': accuracy(y_true, y_pred),
    #     'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    # }
    #
    # precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    # report['precision'] = precision
    # report['recall'] = recall
    # report['f1'] = f1
    #
    # if y_pred_proba is not None and len(np.unique(y_true)) == 2:
    #     report['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    #
    # return report

    pass
```

---

## Composition Example: Binary Classification

```python
# Load data
X, y = load_binary_data()  # y in {0, 1}

# Split and standardize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_scaled, mean, std = standardize(X_train)
X_test_scaled = (X_test - mean) / std

# Initialize
n_features = X_train_scaled.shape[1]
weights = np.zeros(n_features)
bias = 0.0

# Training loop
learning_rate = 0.1
num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass
    y_pred_proba = predict_proba_binary(X_train_scaled, weights, bias)

    # Compute loss
    loss = binary_cross_entropy(y_train, y_pred_proba)

    # Compute gradients
    grad_w, grad_b = binary_cross_entropy_gradient(
        X_train_scaled, y_train, y_pred_proba
    )

    # Update
    weights -= learning_rate * grad_w
    bias -= learning_rate * grad_b

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Evaluate
y_test_proba = predict_proba_binary(X_test_scaled, weights, bias)
y_test_pred = predict_binary(X_test_scaled, weights, bias)

report = classification_report(y_test, y_test_pred, y_test_proba)
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Precision: {report['precision']:.4f}")
print(f"Recall: {report['recall']:.4f}")
print(f"F1: {report['f1']:.4f}")
print(f"ROC AUC: {report['roc_auc']:.4f}")
```

---

## Composition Example: Multi-class Classification

```python
# Load data
X, y = load_multiclass_data()  # y in {0, 1, 2, ...}

# One-hot encode labels
n_classes = len(np.unique(y))
y_onehot = np.eye(n_classes)[y]

# Split and standardize
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2)
X_train_scaled, mean, std = standardize(X_train)
X_test_scaled = (X_test - mean) / std

# Initialize
n_features = X_train_scaled.shape[1]
weights = np.zeros((n_features, n_classes))
bias = np.zeros(n_classes)

# Training loop
learning_rate = 0.1
num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass
    y_pred_proba = predict_proba_multiclass(X_train_scaled, weights, bias)

    # Compute loss
    loss = categorical_cross_entropy(y_train, y_pred_proba)

    # Compute gradients
    grad_w, grad_b = categorical_cross_entropy_gradient(
        X_train_scaled, y_train, y_pred_proba
    )

    # Update
    weights -= learning_rate * grad_w
    bias -= learning_rate * grad_b

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Evaluate
y_test_pred = predict_multiclass(X_test_scaled, weights, bias)
y_test_true_labels = np.argmax(y_test, axis=1)

acc = accuracy(y_test_true_labels, y_test_pred)
cm = confusion_matrix(y_test_true_labels, y_test_pred, n_classes)
precision, recall, f1 = precision_recall_f1(
    y_test_true_labels, y_test_pred, average='macro'
)

print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1 (macro): {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")
```

---

## Constraints

### Allowed
- NumPy: All functions
- Python standard library
- For ROC AUC: You may use `sklearn.metrics.roc_auc_score` for simplicity

### Not Allowed
- Scikit-learn for other functions (except as noted)
- SciPy
- TensorFlow, PyTorch
- Any ML frameworks

### Code Style
- Pure functions (no side effects)
- Type hints for all functions
- Clear, simple implementations
- Organize into 3 files:
  - `activations.py`: Functions 1-6
  - `losses.py`: Functions 7-10
  - `metrics.py`: Functions 11-15

---

## Testing

Your implementation will be tested on:

1. **Correctness**: Each function produces correct outputs
2. **Numerical Stability**: No overflow/underflow
3. **Gradient Accuracy**: Gradients match numerical differentiation
4. **Edge Cases**: Perfect predictions, all same class, empty arrays
5. **Composition**: Functions work together in training loops
6. **Performance**: Loss decreases, metrics improve

Run tests:
```bash
pytest stages/s07_logistic_regression/tests/ -v
python scripts/grade.py s07_logistic_regression
```

---

## Success Criteria

✅ All 15 functions pass unit tests
✅ Sigmoid/softmax numerically stable (no overflow)
✅ Gradients match numerical differentiation (< 1e-5 error)
✅ Training loop decreases loss
✅ Accuracy > 0.8 on test data (for typical datasets)
✅ Binary and multi-class classification both work
✅ Metrics correctly computed (verified against sklearn)

Good luck! Each function is simple - the magic is in composition. 🚀
