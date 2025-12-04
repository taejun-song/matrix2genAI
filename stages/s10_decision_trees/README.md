# Stage 10: Decision Trees & Ensemble Methods

## Overview

Build decision trees and ensemble methods from scratch. Learn how trees make decisions by recursively splitting data, and how combining many weak learners creates powerful models.

**The Big Idea:** A decision tree asks a series of yes/no questions to classify data. Random forests and gradient boosting combine many trees to create robust predictions.

## Learning Philosophy

You will implement:
- Split criteria (entropy, Gini, MSE)
- Recursive tree building (CART algorithm)
- Decision tree classifier and regressor
- Random Forest and Gradient Boosting

**Time:** 8-10 hours
**Difficulty:** ⭐⭐⭐⭐

## Getting Started

```bash
cd stages/s10_decision_trees
uv run pytest tests/ -v
```

### Files You'll Edit

- `starter/split_criteria.py` - Information gain, Gini impurity
- `starter/tree_building.py` - Node splitting and tree construction
- `starter/decision_tree.py` - DecisionTree classes
- `starter/ensemble.py` - RandomForest, GradientBoosting

---

## Conceptual Understanding

### How Trees Make Decisions

```
Is income > $50k?
       │
    ┌──┴──┐
   Yes    No
    │      │
 Credit   Is age > 30?
 score>700?     │
    │       ┌──┴──┐
  ┌─┴─┐   Yes    No
 Yes  No   │      │
  │    │  Approve Deny
Approve Deny
```

### Split Criteria

**Entropy (Information Theory):**
```
H(S) = -Σ pᵢ log₂(pᵢ)

Example: [5 cats, 5 dogs]
H = -0.5 log₂(0.5) - 0.5 log₂(0.5) = 1 bit

After split: [4 cats, 1 dog] | [1 cat, 4 dogs]
H_left = -0.8 log₂(0.8) - 0.2 log₂(0.2) = 0.72
H_right = -0.2 log₂(0.2) - 0.8 log₂(0.8) = 0.72

Information Gain = 1 - (0.5×0.72 + 0.5×0.72) = 0.28
```

**Gini Impurity:**
```
Gini(S) = 1 - Σ pᵢ²

Pure node (all same class): Gini = 0
Most impure (50/50): Gini = 0.5
```

### Random Forest

```
Bagging + Feature Randomness:
1. Create B bootstrap samples
2. For each sample, train a tree with random feature subset
3. Aggregate predictions (voting or averaging)

Why it works:
- Reduces variance through averaging
- Decorrelates trees via feature randomness
- Each tree sees different view of data
```

### Gradient Boosting

```
Sequential Improvement:
1. Fit initial model (often just mean)
2. Compute residuals (errors)
3. Fit new tree to predict residuals
4. Add new tree × learning_rate to ensemble
5. Repeat

F₀(x) = mean(y)
For m = 1 to M:
    rᵢ = yᵢ - Fₘ₋₁(xᵢ)  # residuals
    fit tree hₘ to residuals
    Fₘ(x) = Fₘ₋₁(x) + η × hₘ(x)
```

## What You'll Build

### Split Criteria (5 functions)
1. `entropy(y)` - Shannon entropy
2. `information_gain(y, y_left, y_right)` - IG for splits
3. `gini_impurity(y)` - Gini index
4. `gini_gain(y, y_left, y_right)` - Gini-based gain
5. `mse_reduction(y, y_left, y_right)` - For regression

### Tree Building (4 functions)
6. `find_best_split(X, y, feature_idx, criterion)` - Best threshold
7. `find_best_feature_split(X, y, criterion)` - Best feature + threshold
8. `build_tree(X, y, max_depth, min_samples, criterion)` - Recursive building
9. `predict_sample(tree, x)` - Traverse tree

### Decision Tree Classes (4)
10. `DecisionTreeClassifier` - Classification
11. `DecisionTreeRegressor` - Regression

### Ensemble Methods (6)
12. `bootstrap_sample(X, y)` - Random sampling with replacement
13. `random_subspace(n_features, max_features)` - Feature subset
14. `RandomForestClassifier` - Ensemble of trees
15. `RandomForestRegressor` - Regression ensemble
16. `GradientBoostingRegressor` - Sequential boosting

## Success Criteria

- Trees correctly classify simple datasets
- Information gain matches expected values
- Random forest reduces overfitting
- Gradient boosting improves iteratively

**Target: All tests passing**
