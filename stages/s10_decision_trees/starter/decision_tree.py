from __future__ import annotations

import numpy as np

from stages.s10_decision_trees.starter.tree_building import build_tree, predict_sample


class DecisionTreeClassifier:
    """Decision tree for classification."""

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        criterion: str = "gini",
    ):
        """
        Args:
            max_depth: Maximum depth of tree
            min_samples_split: Minimum samples required to split
            criterion: 'gini' or 'entropy'
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """
        Build decision tree from training data.

        Args:
            X: Features, shape (n_samples, n_features)
            y: Labels, shape (n_samples,)

        Returns:
            self
        """
        # TODO:
        # self.classes_ = np.unique(y)
        # self.tree = build_tree(X, y, self.max_depth, self.min_samples_split, self.criterion)
        # return self
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features, shape (n_samples, n_features)

        Returns:
            predictions: Predicted classes, shape (n_samples,)
        """
        # TODO:
        # return np.array([predict_sample(self.tree, x) for x in X])
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (for leaf node class distribution).

        Args:
            X: Features

        Returns:
            probabilities: Shape (n_samples, n_classes)
        """
        raise NotImplementedError


class DecisionTreeRegressor:
    """Decision tree for regression."""

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        """Build regression tree."""
        # TODO:
        # self.tree = build_tree(X, y, self.max_depth, self.min_samples_split, 'mse')
        # return self
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values."""
        # TODO:
        # return np.array([predict_sample(self.tree, x) for x in X])
        raise NotImplementedError
