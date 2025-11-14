# Stage 7: Logistic Regression & Classification - Building Blocks

## Overview

Build a complete classification system from simple, composable functions. Just like linear regression, but for predicting categories instead of numbers.

**The Big Idea:** Logistic regression transforms linear regression's continuous outputs into probabilities, allowing us to classify data into categories. It's the foundation for neural networks!

## Learning Philosophy

You will implement **small building blocks** (1-3 lines each) that compose into complete algorithms:
- Binary and multi-class classifiers
- Training loops with cross-entropy loss
- Classification metrics and evaluation
- Decision boundary visualization

**Time:** 3-4 hours
**Difficulty:** ⭐⭐⭐

## Conceptual Understanding

### Why Not Just Use Linear Regression for Classification?

Linear regression outputs can be anything: -1000, 3.7, 1000000. But for classification, we need:
- **Probabilities**: values between 0 and 1
- **Interpretability**: "70% chance this email is spam"
- **Decision boundaries**: clear separation between classes

**Example Problem:**
```
Email classification: spam (1) or not spam (0)

Bad (linear regression):
  ŷ = Xw + b = -2.5  ❌  What does -2.5 mean?
  ŷ = Xw + b = 3.7   ❌  Probability > 1?

Good (logistic regression):
  p = σ(Xw + b) = 0.08  ✓  8% chance of spam → not spam
  p = σ(Xw + b) = 0.95  ✓  95% chance of spam → spam!
```

### The Sigmoid Function: From Linear to Probability

The sigmoid function σ(z) = 1/(1+e^(-z)) is the magic bridge:

```
Input (linear):  z = -∞ ... -2 ... 0 ... 2 ... +∞
                     ↓       ↓     ↓     ↓      ↓
Sigmoid output:  p =  0    0.12  0.5  0.88    1

Properties:
  • S-shaped curve (smooth transition)
  • Always outputs between 0 and 1
  • σ(0) = 0.5 (neutral decision boundary)
  • σ(large positive) ≈ 1 (confident "yes")
  • σ(large negative) ≈ 0 (confident "no")
```

**Intuition:** Think of sigmoid as a "soft threshold" - instead of a hard cutoff at 0 (like `z > 0 ? 1 : 0`), it gradually transitions from 0 to 1, giving us confidence levels.

## What You'll Build

### Core Building Blocks (10 functions)

1. `sigmoid(z)` - Squash values to [0, 1]
2. `softmax(z)` - Multi-class probabilities
3. `predict_proba_binary(X, weights, bias)` - Binary class probabilities
4. `predict_proba_multiclass(X, weights, bias)` - Multi-class probabilities
5. `predict_binary(X, weights, bias, threshold)` - Binary predictions
6. `predict_multiclass(X, weights, bias)` - Multi-class predictions
7. `binary_cross_entropy(y_true, y_pred_proba)` - Binary loss
8. `categorical_cross_entropy(y_true, y_pred_proba)` - Multi-class loss
9. `binary_cross_entropy_gradient(X, y_true, y_pred_proba)` - Binary gradients
10. `categorical_cross_entropy_gradient(X, y_true, y_pred_proba)` - Multi-class gradients

### Evaluation Metrics (5 functions)

11. `accuracy(y_true, y_pred)` - Fraction correct
12. `confusion_matrix(y_true, y_pred)` - Classification table
13. `precision_recall_f1(y_true, y_pred)` - Classification quality
14. `roc_auc_score(y_true, y_pred_proba)` - Ranking quality
15. `classification_report(y_true, y_pred, y_pred_proba)` - Complete summary

### What You'll Compose

Using these blocks, you'll write:
- Binary classification training loop
- Multi-class classification (one-vs-rest)
- Complete evaluation pipelines
- Decision boundary visualizations

## Mathematical Background

### Understanding Cross-Entropy Loss

**Why not use MSE for classification?**

Mean Squared Error penalizes all errors equally:
```
Wrong prediction with MSE:
  y=1, p=0.1  → (1-0.1)² = 0.81
  y=1, p=0.9  → (1-0.9)² = 0.01

Problem: MSE doesn't care much about being "confidently wrong"
  Predicting 0.1 when answer is 1 should be heavily penalized!
```

**Cross-Entropy: Confidence-Aware Loss**

Cross-entropy penalizes confident mistakes harshly:
```
Binary Cross-Entropy:
  y=1, p=0.9  → -log(0.9)   = 0.11  ✓ Good, low loss
  y=1, p=0.5  → -log(0.5)   = 0.69
  y=1, p=0.1  → -log(0.1)   = 2.30  ❌ Bad, high loss!
  y=1, p=0.01 → -log(0.01)  = 4.61  ❌ VERY BAD, huge loss!

Intuition: -log(p) shoots to infinity as p→0
  Being confidently wrong (p=0.01 when y=1) is heavily punished!
```

**The Full Formula:**
```
L = -(1/n) Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]

When y=1: L = -log(p)     (want p close to 1)
When y=0: L = -log(1-p)   (want p close to 0)
```

### From Binary to Multi-Class: Softmax

**Sigmoid generalizes to Softmax for K classes:**

```
Binary (2 classes):
  p(class 1) = σ(z) = 1/(1+e^(-z))
  p(class 0) = 1 - σ(z)

Multi-class (K classes):
  p(class j) = e^(zⱼ) / Σₖ e^(zₖ)   for j=1,2,...,K

Example with 3 classes (cat, dog, bird):
  z = [2.0, 1.0, 0.1]  (raw scores)
        ↓ softmax
  p = [0.66, 0.24, 0.10]  (probabilities sum to 1)
  Prediction: cat (highest probability)
```

**Why exponential?**
- Ensures all outputs are positive
- Amplifies differences (2.0 vs 1.0 → 0.66 vs 0.24, clear winner)
- Differentiable (can use gradient descent)

### The Gradient Miracle

**Amazing fact:** Despite using sigmoid/softmax, the gradient has the same simple form!

```
For both binary and multi-class:
  ∂L/∂w = (1/n) Xᵀ(predicted - actual)

This is the SAME as linear regression gradient!
  The sigmoid/softmax cancels out when we take derivatives
  of the cross-entropy loss. Mathematical beauty!
```

---

## Detailed Mathematical Formulas

### Logistic Regression (Binary)

```
z = Xw + b                    (linear combination)
σ(z) = 1 / (1 + e^(-z))      (sigmoid activation)
P(y=1|X) = σ(Xw + b)         (probability of class 1)

where:
  X ∈ ℝⁿˣᵈ  (n samples, d features)
  w ∈ ℝᵈ    (weights)
  b ∈ ℝ     (bias)
  σ(z) ∈ (0, 1)  (probability)
```

### Sigmoid Function

```
σ(z) = 1 / (1 + e^(-z))

Properties:
  σ(0) = 0.5
  σ(+∞) = 1.0
  σ(−∞) = 0.0
  σ'(z) = σ(z)(1 - σ(z))  (convenient derivative!)
```

### Binary Cross-Entropy Loss

```
L(w,b) = -(1/n) Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]

where:
  yᵢ ∈ {0, 1}        (true label)
  pᵢ = σ(Xᵢw + b)    (predicted probability)
```

### Gradients (Binary)

```
∂L/∂w = (1/n) Xᵀ(p - y)
∂L/∂b = (1/n) Σᵢ(pᵢ - yᵢ)

Remarkably similar to linear regression!
```

### Softmax (Multi-class)

```
softmax(z)ⱼ = e^(zⱼ) / Σₖ e^(zₖ)

Properties:
  Σⱼ softmax(z)ⱼ = 1       (probabilities sum to 1)
  softmax(z)ⱼ ∈ (0, 1)     (each is a valid probability)
  max_arg softmax(z) = arg max(z)  (preserves ordering)
```

### Categorical Cross-Entropy Loss

```
L(W,b) = -(1/n) Σᵢ Σⱼ yᵢⱼ log(pᵢⱼ)

where:
  yᵢⱼ ∈ {0, 1}              (one-hot encoded labels)
  pᵢⱼ = softmax(XᵢW + b)ⱼ   (predicted probability for class j)
  W ∈ ℝᵈˣᶜ                  (weight matrix for c classes)
```

### Gradients (Multi-class)

```
∂L/∂W = (1/n) Xᵀ(P - Y)
∂L/∂b = (1/n) Σᵢ(pᵢ - yᵢ)

where:
  P ∈ ℝⁿˣᶜ  (predicted probabilities)
  Y ∈ ℝⁿˣᶜ  (one-hot encoded labels)
```

## Implementation Guide

### Step 1: Activation Functions (15 min)

Implement sigmoid and softmax:

**Example: `sigmoid`**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # That's it!
```

**Example: `softmax`**
```python
def softmax(z):
    # Numerical stability: subtract max
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
```

### Step 2: Prediction Functions (20 min)

Chain activations with linear models:

```python
def predict_proba_binary(X, weights, bias):
    z = X @ weights + bias
    return sigmoid(z)  # Compose!

def predict_binary(X, weights, bias, threshold=0.5):
    proba = predict_proba_binary(X, weights, bias)
    return (proba >= threshold).astype(int)
```

### Step 3: Loss Functions (20 min)

Implement cross-entropy losses:

```python
def binary_cross_entropy(y_true, y_pred_proba):
    # Clip to avoid log(0)
    eps = 1e-15
    p = np.clip(y_pred_proba, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
```

### Step 4: Gradients (25 min)

Derive and implement gradient functions:

```python
def binary_cross_entropy_gradient(X, y_true, y_pred_proba):
    n = len(y_true)
    grad_w = (1/n) * X.T @ (y_pred_proba - y_true)
    grad_b = (1/n) * np.sum(y_pred_proba - y_true)
    return grad_w, grad_b
```

### Step 5: Metrics (30 min)

Implement evaluation metrics:

```python
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    # For binary: [[TN, FP], [FN, TP]]
    # Returns 2D array
    pass  # Your implementation
```

### Step 6: Compose Training Loop (45 min)

Write gradient descent for classification:

```python
# Initialize
weights = np.zeros(n_features)
bias = 0.0
learning_rate = 0.1

# Train
for epoch in range(1000):
    # Forward pass
    y_pred_proba = predict_proba_binary(X_train, weights, bias)

    # Compute loss
    loss = binary_cross_entropy(y_train, y_pred_proba)

    # Compute gradients
    grad_w, grad_b = binary_cross_entropy_gradient(
        X_train, y_train, y_pred_proba
    )

    # Update parameters
    weights -= learning_rate * grad_w
    bias -= learning_rate * grad_b

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Evaluate
y_pred = predict_binary(X_test, weights, bias)
acc = accuracy(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
```

## Key Concepts

### Why Sigmoid for Binary Classification?

Linear regression outputs unbounded values: `ŷ ∈ (-∞, +∞)`

But we need probabilities: `P(y=1) ∈ [0, 1]`

Sigmoid does the mapping:
```
z = -10  →  σ(z) ≈ 0.00005  (very confident class 0)
z = 0    →  σ(z) = 0.5      (uncertain)
z = +10  →  σ(z) ≈ 0.99995  (very confident class 1)
```

### Why Cross-Entropy Loss?

MSE loss for classification is problematic:
```
# When prediction is wrong (y=1, p=0.1):
MSE gradient ∝ (p - y) * σ'(z) = (-0.9) * 0.09 = -0.081  (small!)

# Cross-entropy gradient:
CE gradient ∝ (p - y) = -0.9  (much larger!)
```

Cross-entropy has better gradients when predictions are wrong!

### Decision Boundary

The decision boundary is where `P(y=1) = 0.5`:

```
σ(Xw + b) = 0.5
Xw + b = 0        (since σ(0) = 0.5)

This is a hyperplane in feature space!
```

For 2D (2 features):
```
w₁x₁ + w₂x₂ + b = 0
x₂ = -(w₁/w₂)x₁ - b/w₂  (line equation)
```

### Multi-class Classification

**One-vs-Rest (OvR):**
- Train K binary classifiers (one per class)
- Predict class with highest confidence

**Softmax:**
- Train single model with K outputs
- Softmax ensures probabilities sum to 1
- More efficient and consistent

## Common Pitfalls

### 1. Numerical Instability in Sigmoid

```python
# ❌ Bad: Overflow for large negative z
def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # exp(-1000) = inf!

# ✅ Good: Clip inputs
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Avoid overflow
    return 1 / (1 + np.exp(-z))
```

### 2. Log of Zero in Cross-Entropy

```python
# ❌ Bad: log(0) = -inf
loss = -np.mean(y * np.log(p))  # If p=0, crash!

# ✅ Good: Clip probabilities
eps = 1e-15
p = np.clip(p, eps, 1 - eps)
loss = -np.mean(y * np.log(p))
```

### 3. Softmax Overflow

```python
# ❌ Bad: exp(1000) = inf
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

# ✅ Good: Subtract max (numerically stable)
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
```

### 4. Wrong Threshold

```python
# Default threshold is 0.5, but it's not always optimal!

# For imbalanced classes (95% negative, 5% positive):
y_pred = predict_binary(X, w, b, threshold=0.5)  # Predicts mostly 0
accuracy = 0.95  # Looks good but useless!

# Better: Tune threshold
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    y_pred = predict_binary(X, w, b, threshold)
    f1 = f1_score(y_test, y_pred)
    print(f"Threshold {threshold}: F1 = {f1:.4f}")
```

### 5. Not One-Hot Encoding for Multi-class

```python
# ❌ Bad: Integer labels with softmax loss
y = np.array([0, 1, 2, 1, 0])  # Class indices
loss = categorical_cross_entropy(y, probs)  # Wrong!

# ✅ Good: One-hot encode first
y_onehot = np.eye(3)[y]  # [[1,0,0], [0,1,0], [0,0,1], ...]
loss = categorical_cross_entropy(y_onehot, probs)
```

## Classification Metrics Explained

### Confusion Matrix (Binary)

```
                Predicted
                0      1
Actual  0    [ TN  |  FP ]
        1    [ FN  |  TP ]

TN = True Negatives  (correct rejections)
FP = False Positives (false alarms)
FN = False Negatives (misses)
TP = True Positives  (hits)
```

### Precision and Recall

```
Precision = TP / (TP + FP)  "When I predict positive, how often am I right?"
Recall    = TP / (TP + FN)  "Of all actual positives, how many did I catch?"
F1        = 2 * (Precision * Recall) / (Precision + Recall)  (harmonic mean)
```

**Example: Spam Detection**
- High Precision: Few false alarms (important emails not marked spam)
- High Recall: Catch all spam (don't miss any spam)

### ROC-AUC Score

ROC curve plots:
- X-axis: False Positive Rate = FP / (FP + TN)
- Y-axis: True Positive Rate = TP / (TP + FN) = Recall

AUC = Area Under ROC Curve
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing
- AUC < 0.5: Worse than random (predictions are inverted!)

## Experiments to Try

### 1. Threshold Tuning

```python
thresholds = np.linspace(0.1, 0.9, 9)
for t in thresholds:
    y_pred = predict_binary(X_test, w, b, threshold=t)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Threshold {t:.1f}: P={precision:.3f}, R={recall:.3f}")
```

### 2. Multi-class on Iris Dataset

```python
# Load iris (3 classes, 4 features)
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

# Train softmax classifier
# Compare OvR vs softmax approach
```

### 3. Decision Boundary Visualization

```python
# For 2D features
import matplotlib.pyplot as plt

# Create grid
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
    np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
)
grid = np.c_[xx.ravel(), yy.ravel()]

# Predict on grid
Z = predict_proba_binary(grid, weights, bias)
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
```

## Debugging Guide

### Predictions are all one class

```python
# Check class balance
print(f"Class 0: {np.sum(y_train == 0)} samples")
print(f"Class 1: {np.sum(y_train == 1)} samples")

# Check predicted probabilities
probs = predict_proba_binary(X_train, weights, bias)
print(f"Prob range: [{probs.min():.3f}, {probs.max():.3f}]")

# If all probs ≈ 0.5, weights might be too small
print(f"Weight magnitude: {np.linalg.norm(weights):.3f}")

# Try larger learning rate or more epochs
```

### Loss is NaN

```python
# Check for overflow in probabilities
probs = predict_proba_binary(X_train, weights, bias)
print(f"Any NaN in probs? {np.any(np.isnan(probs))}")
print(f"Any inf in probs? {np.any(np.isinf(probs))}")

# Check input z values
z = X_train @ weights + bias
print(f"z range: [{z.min():.3f}, {z.max():.3f}]")

# If |z| > 100, inputs might need standardization
X_train_scaled, mean, std = standardize(X_train)
```

### Loss not decreasing

```python
# 1. Check gradient correctness
from stages.s03_calculus.starter.numerical_diff import gradient_check

def loss_fn(w):
    proba = predict_proba_binary(X_train, w, bias)
    return binary_cross_entropy(y_train, proba)

proba = predict_proba_binary(X_train, weights, bias)
grad_analytical, _ = binary_cross_entropy_gradient(X_train, y_train, proba)
is_correct = gradient_check(loss_fn, weights, grad_analytical)
print(f"Gradient correct: {is_correct}")

# 2. Learning rate too large?
learning_rate = 0.01  # Try smaller

# 3. Standardize features
X_train_scaled, mean, std = standardize(X_train)
```

## Testing Your Implementation

```bash
# Test activation functions
pytest tests/test_activations.py -v

# Test loss functions and gradients
pytest tests/test_losses.py -v

# Test metrics
pytest tests/test_metrics.py -v

# Test full binary classification
pytest tests/test_binary_classification.py -v

# Test multi-class classification
pytest tests/test_multiclass_classification.py -v

# Grade your work
python scripts/grade.py s07_logistic_regression
```

## Real-World Applications

Logistic regression is used daily in industry:

- **Spam Detection**: Email classification (spam vs ham)
- **Medical Diagnosis**: Disease prediction from symptoms
- **Fraud Detection**: Identify fraudulent transactions
- **Click Prediction**: Ad click-through rate estimation
- **Credit Scoring**: Loan approval decisions
- **Customer Churn**: Predict if customer will leave

Why it's still popular in 2024:
- Fast to train (even on millions of samples)
- Interpretable (feature coefficients show importance)
- Probabilistic outputs (useful for decision-making)
- Works well with proper feature engineering
- Strong baseline for comparison

## What's Next

After mastering logistic regression:

**s08: Feature Engineering** - Encoding, imputation, selection
**s09: Regularization** - L1/L2 for logistic regression
**s11: Neural Networks** - Stack logistic units with hidden layers
**s12: Softmax Classifier** - Deep learning's final layer

The building blocks stay the same:
1. Simple functions
2. Test independently
3. Compose into systems

## Success Criteria

You understand this stage when you can:

- ✅ Explain why sigmoid maps to probabilities
- ✅ Derive cross-entropy gradient from scratch
- ✅ Write binary and multi-class training loops
- ✅ Debug numerical stability issues
- ✅ Interpret classification metrics for business decisions
- ✅ Tune decision threshold for different use cases

**Target: 90%+ test passing rate**

Good luck! Classification is one of the most practical ML tasks. Master these building blocks and you'll use them forever. 🚀
