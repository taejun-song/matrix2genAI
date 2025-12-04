# Stage 10: Decision Trees - Specification

## Building Blocks to Implement

### Module 1: Split Criteria

```python
def entropy(y: np.ndarray) -> float:
    """Shannon entropy: H(S) = -Σ pᵢ log₂(pᵢ)"""
    # TODO: Compute class probabilities and entropy
    raise NotImplementedError

def information_gain(y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """IG = H(parent) - weighted_avg(H(children))"""
    # TODO: Compute weighted entropy reduction
    raise NotImplementedError

def gini_impurity(y: np.ndarray) -> float:
    """Gini = 1 - Σ pᵢ²"""
    # TODO: Compute Gini impurity
    raise NotImplementedError

def gini_gain(y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Gini gain for split"""
    raise NotImplementedError

def mse_reduction(y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """MSE reduction for regression trees"""
    raise NotImplementedError
```

### Module 2: Tree Building

```python
def find_best_split(
    X: np.ndarray, y: np.ndarray, feature_idx: int, criterion: str
) -> tuple[float, float]:
    """Find best threshold for a single feature. Returns (threshold, gain)."""
    raise NotImplementedError

def find_best_feature_split(
    X: np.ndarray, y: np.ndarray, criterion: str
) -> tuple[int, float, float]:
    """Find best feature and threshold. Returns (feature_idx, threshold, gain)."""
    raise NotImplementedError

def build_tree(
    X: np.ndarray, y: np.ndarray,
    max_depth: int, min_samples: int, criterion: str, depth: int = 0
) -> dict:
    """Recursively build decision tree. Returns tree as nested dict."""
    raise NotImplementedError

def predict_sample(tree: dict, x: np.ndarray) -> float:
    """Traverse tree to predict single sample."""
    raise NotImplementedError
```

### Module 3: Decision Tree Classes

```python
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
```

### Module 4: Ensemble Methods

```python
def bootstrap_sample(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create bootstrap sample (sample with replacement)."""
    raise NotImplementedError

def random_subspace(n_features: int, max_features: int | str) -> np.ndarray:
    """Select random subset of features."""
    raise NotImplementedError

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt'):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingRegressor':
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
```

## Testing

```bash
pytest stages/s10_decision_trees/tests/ -v
```
