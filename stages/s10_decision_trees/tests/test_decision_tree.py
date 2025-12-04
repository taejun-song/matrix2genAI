from __future__ import annotations

import numpy as np

from stages.s10_decision_trees.starter.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


class TestDecisionTreeClassifier:
    def test_fit_predict(self) -> None:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
        clf = DecisionTreeClassifier(max_depth=2)
        clf.fit(X, y)
        predictions = clf.predict(X)
        np.testing.assert_array_equal(predictions, y)

    def test_pure_class(self) -> None:
        X = np.array([[1], [2], [3]])
        y = np.array([0, 0, 0])
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        assert np.all(clf.predict(X) == 0)

    def test_max_depth(self) -> None:
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(X, y)
        accuracy = np.mean(clf.predict(X) == y)
        assert accuracy > 0.5


class TestDecisionTreeRegressor:
    def test_fit_predict(self) -> None:
        X = np.array([[1], [2], [3], [4]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        reg = DecisionTreeRegressor(max_depth=3)
        reg.fit(X, y)
        predictions = reg.predict(X)
        mse = np.mean((predictions - y) ** 2)
        assert mse < 1.0

    def test_constant_target(self) -> None:
        X = np.array([[1], [2], [3]])
        y = np.array([5.0, 5.0, 5.0])
        reg = DecisionTreeRegressor()
        reg.fit(X, y)
        np.testing.assert_allclose(reg.predict(X), [5.0, 5.0, 5.0])
