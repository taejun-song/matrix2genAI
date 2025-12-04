from __future__ import annotations

import numpy as np

from stages.s10_decision_trees.starter.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


def bootstrap_sample(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create bootstrap sample (sample with replacement).

    Args:
        X: Features, shape (n_samples, n_features)
        y: Labels, shape (n_samples,)

    Returns:
        X_bootstrap, y_bootstrap: Bootstrap samples
    """
    # TODO:
    # n_samples = len(X)
    # indices = np.random.choice(n_samples, size=n_samples, replace=True)
    # return X[indices], y[indices]
    raise NotImplementedError


def random_subspace(n_features: int, max_features: int | str) -> np.ndarray:
    """
    Select random subset of features.

    Args:
        n_features: Total number of features
        max_features: Number of features or 'sqrt', 'log2'

    Returns:
        feature_indices: Array of selected feature indices
    """
    # TODO:
    # if max_features == 'sqrt':
    #     k = int(np.sqrt(n_features))
    # elif max_features == 'log2':
    #     k = int(np.log2(n_features))
    # else:
    #     k = max_features
    # k = max(1, min(k, n_features))
    # return np.random.choice(n_features, size=k, replace=False)
    raise NotImplementedError


class RandomForestClassifier:
    """Random Forest for classification."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: int | str = "sqrt",
        min_samples_split: int = 2,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.trees = []
        self.feature_indices = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """
        Build forest of trees.

        Args:
            X: Features
            y: Labels

        Returns:
            self
        """
        # TODO:
        # self.trees = []
        # self.feature_indices = []
        # for _ in range(self.n_estimators):
        #     X_boot, y_boot = bootstrap_sample(X, y)
        #     feat_idx = random_subspace(X.shape[1], self.max_features)
        #     tree = DecisionTreeClassifier(
        #         max_depth=self.max_depth,
        #         min_samples_split=self.min_samples_split
        #     )
        #     tree.fit(X_boot[:, feat_idx], y_boot)
        #     self.trees.append(tree)
        #     self.feature_indices.append(feat_idx)
        # return self
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict by majority voting.

        Args:
            X: Features

        Returns:
            predictions: Majority vote predictions
        """
        # TODO:
        # predictions = np.array([
        #     tree.predict(X[:, feat_idx])
        #     for tree, feat_idx in zip(self.trees, self.feature_indices)
        # ])
        # # Majority vote
        # return np.array([
        #     np.bincount(predictions[:, i]).argmax()
        #     for i in range(X.shape[0])
        # ])
        raise NotImplementedError


class RandomForestRegressor:
    """Random Forest for regression."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: int | str = "sqrt",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        """Build forest of regression trees."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by averaging tree predictions."""
        raise NotImplementedError


class GradientBoostingRegressor:
    """Gradient Boosting for regression."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        """
        Fit gradient boosting model.

        Algorithm:
            1. Initial prediction = mean(y)
            2. For each iteration:
               - Compute residuals
               - Fit tree to residuals
               - Update predictions
        """
        # TODO:
        # self.initial_prediction = np.mean(y)
        # predictions = np.full(len(y), self.initial_prediction)
        #
        # for _ in range(self.n_estimators):
        #     residuals = y - predictions
        #     tree = DecisionTreeRegressor(max_depth=self.max_depth)
        #     tree.fit(X, residuals)
        #     self.trees.append(tree)
        #     predictions += self.learning_rate * tree.predict(X)
        #
        # return self
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict by summing tree predictions.

        Args:
            X: Features

        Returns:
            predictions: Initial + sum(learning_rate * tree_predictions)
        """
        # TODO:
        # predictions = np.full(len(X), self.initial_prediction)
        # for tree in self.trees:
        #     predictions += self.learning_rate * tree.predict(X)
        # return predictions
        raise NotImplementedError
