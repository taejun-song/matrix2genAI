from __future__ import annotations

import numpy as np

from stages.s10_decision_trees.starter.split_criteria import (
    gini_gain,
    information_gain,
    mse_reduction,
)


def find_best_split(
    X: np.ndarray, y: np.ndarray, feature_idx: int, criterion: str
) -> tuple[float, float]:
    """
    Find best threshold for a single feature.

    Args:
        X: Features, shape (n_samples, n_features)
        y: Labels, shape (n_samples,)
        feature_idx: Index of feature to split on
        criterion: 'gini', 'entropy', or 'mse'

    Returns:
        threshold: Best split threshold
        gain: Gain achieved by this split
    """
    # TODO:
    # gain_fn = {'gini': gini_gain, 'entropy': information_gain, 'mse': mse_reduction}[criterion]
    # feature_values = X[:, feature_idx]
    # thresholds = np.unique(feature_values)
    #
    # best_threshold, best_gain = None, -np.inf
    # for threshold in thresholds:
    #     left_mask = feature_values <= threshold
    #     right_mask = ~left_mask
    #     if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
    #         continue
    #     gain = gain_fn(y, y[left_mask], y[right_mask])
    #     if gain > best_gain:
    #         best_gain = gain
    #         best_threshold = threshold
    # return best_threshold, best_gain
    raise NotImplementedError


def find_best_feature_split(
    X: np.ndarray, y: np.ndarray, criterion: str
) -> tuple[int, float, float]:
    """
    Find best feature and threshold for splitting.

    Args:
        X: Features
        y: Labels
        criterion: Split criterion

    Returns:
        feature_idx: Best feature index
        threshold: Best threshold
        gain: Best gain
    """
    # TODO:
    # best_feature, best_threshold, best_gain = None, None, -np.inf
    # for feature_idx in range(X.shape[1]):
    #     threshold, gain = find_best_split(X, y, feature_idx, criterion)
    #     if threshold is not None and gain > best_gain:
    #         best_gain = gain
    #         best_threshold = threshold
    #         best_feature = feature_idx
    # return best_feature, best_threshold, best_gain
    raise NotImplementedError


def build_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int | None,
    min_samples: int,
    criterion: str,
    depth: int = 0,
) -> dict:
    """
    Recursively build decision tree.

    Args:
        X: Features
        y: Labels
        max_depth: Maximum tree depth (None = unlimited)
        min_samples: Minimum samples to split
        criterion: Split criterion
        depth: Current depth

    Returns:
        tree: Nested dict with 'feature', 'threshold', 'left', 'right', 'value'
    """
    # TODO:
    # # Base cases
    # if len(np.unique(y)) == 1:  # Pure node
    #     return {'value': y[0]}
    # if len(y) < min_samples:
    #     return {'value': np.mean(y) if criterion == 'mse' else np.bincount(y).argmax()}
    # if max_depth is not None and depth >= max_depth:
    #     return {'value': np.mean(y) if criterion == 'mse' else np.bincount(y).argmax()}
    #
    # feature_idx, threshold, gain = find_best_feature_split(X, y, criterion)
    # if feature_idx is None or gain <= 0:
    #     return {'value': np.mean(y) if criterion == 'mse' else np.bincount(y).argmax()}
    #
    # left_mask = X[:, feature_idx] <= threshold
    # return {
    #     'feature': feature_idx,
    #     'threshold': threshold,
    #     'left': build_tree(X[left_mask], y[left_mask], max_depth, min_samples, criterion, depth+1),
    #     'right': build_tree(X[~left_mask], y[~left_mask], max_depth, min_samples, criterion, depth+1),
    # }
    raise NotImplementedError


def predict_sample(tree: dict, x: np.ndarray) -> float:
    """
    Traverse tree to predict single sample.

    Args:
        tree: Decision tree (nested dict)
        x: Single sample, shape (n_features,)

    Returns:
        prediction: Predicted value or class
    """
    # TODO:
    # if 'value' in tree:
    #     return tree['value']
    # if x[tree['feature']] <= tree['threshold']:
    #     return predict_sample(tree['left'], x)
    # return predict_sample(tree['right'], x)
    raise NotImplementedError
