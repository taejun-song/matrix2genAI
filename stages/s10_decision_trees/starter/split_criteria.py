from __future__ import annotations

import numpy as np


def entropy(y: np.ndarray) -> float:
    """
    Compute Shannon entropy: H(S) = -Σ pᵢ log₂(pᵢ)

    Args:
        y: Class labels, shape (n_samples,)

    Returns:
        entropy: Entropy value (0 = pure, higher = more mixed)

    Example:
        >>> entropy(np.array([0, 0, 1, 1]))
        1.0
    """
    # TODO:
    # if len(y) == 0:
    #     return 0.0
    # _, counts = np.unique(y, return_counts=True)
    # probs = counts / len(y)
    # return -np.sum(probs * np.log2(probs + 1e-10))
    raise NotImplementedError


def information_gain(
    y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray
) -> float:
    """
    Compute information gain: IG = H(parent) - weighted_avg(H(children))

    Args:
        y: Parent labels
        y_left: Left child labels
        y_right: Right child labels

    Returns:
        gain: Information gain (higher = better split)

    Example:
        >>> y = np.array([0, 0, 1, 1])
        >>> information_gain(y, np.array([0, 0]), np.array([1, 1]))
        1.0
    """
    # TODO:
    # n = len(y)
    # n_left, n_right = len(y_left), len(y_right)
    # if n_left == 0 or n_right == 0:
    #     return 0.0
    # parent_entropy = entropy(y)
    # child_entropy = (n_left/n) * entropy(y_left) + (n_right/n) * entropy(y_right)
    # return parent_entropy - child_entropy
    raise NotImplementedError


def gini_impurity(y: np.ndarray) -> float:
    """
    Compute Gini impurity: Gini = 1 - Σ pᵢ²

    Args:
        y: Class labels

    Returns:
        gini: Gini impurity (0 = pure, 0.5 = most impure for binary)

    Example:
        >>> gini_impurity(np.array([0, 0, 1, 1]))
        0.5
    """
    # TODO:
    # if len(y) == 0:
    #     return 0.0
    # _, counts = np.unique(y, return_counts=True)
    # probs = counts / len(y)
    # return 1 - np.sum(probs ** 2)
    raise NotImplementedError


def gini_gain(
    y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray
) -> float:
    """
    Compute Gini gain for a split.

    Args:
        y: Parent labels
        y_left: Left child labels
        y_right: Right child labels

    Returns:
        gain: Gini gain
    """
    # TODO:
    # n = len(y)
    # n_left, n_right = len(y_left), len(y_right)
    # if n_left == 0 or n_right == 0:
    #     return 0.0
    # parent_gini = gini_impurity(y)
    # child_gini = (n_left/n) * gini_impurity(y_left) + (n_right/n) * gini_impurity(y_right)
    # return parent_gini - child_gini
    raise NotImplementedError


def mse_reduction(
    y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray
) -> float:
    """
    Compute MSE reduction for regression trees.

    Args:
        y: Parent values
        y_left: Left child values
        y_right: Right child values

    Returns:
        reduction: MSE reduction (higher = better split)
    """
    # TODO:
    # def mse(arr):
    #     if len(arr) == 0:
    #         return 0.0
    #     return np.mean((arr - np.mean(arr)) ** 2)
    # n = len(y)
    # n_left, n_right = len(y_left), len(y_right)
    # if n_left == 0 or n_right == 0:
    #     return 0.0
    # parent_mse = mse(y)
    # child_mse = (n_left/n) * mse(y_left) + (n_right/n) * mse(y_right)
    # return parent_mse - child_mse
    raise NotImplementedError
