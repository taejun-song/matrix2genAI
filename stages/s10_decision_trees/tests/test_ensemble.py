from __future__ import annotations

import numpy as np

from stages.s10_decision_trees.starter.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    bootstrap_sample,
    random_subspace,
)


class TestBootstrapSample:
    def test_same_size(self) -> None:
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        X_boot, y_boot = bootstrap_sample(X, y)
        assert X_boot.shape == X.shape
        assert y_boot.shape == y.shape

    def test_has_duplicates(self) -> None:
        np.random.seed(42)
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        X_boot, _ = bootstrap_sample(X, y)
        assert len(np.unique(X_boot)) < len(X)


class TestRandomSubspace:
    def test_sqrt(self) -> None:
        n_features = 100
        features = random_subspace(n_features, 'sqrt')
        assert len(features) == 10

    def test_log2(self) -> None:
        n_features = 64
        features = random_subspace(n_features, 'log2')
        assert len(features) == 6

    def test_int(self) -> None:
        features = random_subspace(10, 5)
        assert len(features) == 5


class TestRandomForestClassifier:
    def test_fit_predict(self) -> None:
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        rf = RandomForestClassifier(n_estimators=10, max_depth=3)
        rf.fit(X, y)
        predictions = rf.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.7


class TestGradientBoostingRegressor:
    def test_fit_predict(self) -> None:
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
        gb = GradientBoostingRegressor(n_estimators=20, max_depth=2, learning_rate=0.1)
        gb.fit(X, y)
        predictions = gb.predict(X)
        mse = np.mean((predictions - y) ** 2)
        assert mse < 1.0

    def test_improves_over_iterations(self) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] ** 2 + X[:, 1]
        gb_small = GradientBoostingRegressor(n_estimators=5, max_depth=2)
        gb_large = GradientBoostingRegressor(n_estimators=50, max_depth=2)
        gb_small.fit(X, y)
        gb_large.fit(X, y)
        mse_small = np.mean((gb_small.predict(X) - y) ** 2)
        mse_large = np.mean((gb_large.predict(X) - y) ** 2)
        assert mse_large <= mse_small
