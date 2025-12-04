from __future__ import annotations

import numpy as np

from stages.s10_decision_trees.starter.split_criteria import (
    entropy,
    gini_gain,
    gini_impurity,
    information_gain,
    mse_reduction,
)


class TestEntropy:
    def test_pure_class(self) -> None:
        y = np.array([0, 0, 0, 0])
        assert entropy(y) == 0.0

    def test_binary_balanced(self) -> None:
        y = np.array([0, 0, 1, 1])
        np.testing.assert_allclose(entropy(y), 1.0, rtol=1e-5)

    def test_empty(self) -> None:
        y = np.array([])
        assert entropy(y) == 0.0

    def test_multiclass(self) -> None:
        y = np.array([0, 0, 1, 1, 2, 2])
        result = entropy(y)
        expected = -3 * (1/3) * np.log2(1/3)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestInformationGain:
    def test_perfect_split(self) -> None:
        y = np.array([0, 0, 1, 1])
        y_left = np.array([0, 0])
        y_right = np.array([1, 1])
        assert information_gain(y, y_left, y_right) == 1.0

    def test_no_improvement(self) -> None:
        y = np.array([0, 0, 1, 1])
        y_left = np.array([0, 1])
        y_right = np.array([0, 1])
        np.testing.assert_allclose(information_gain(y, y_left, y_right), 0.0, atol=1e-5)

    def test_empty_child(self) -> None:
        y = np.array([0, 0, 1, 1])
        y_left = np.array([])
        y_right = np.array([0, 0, 1, 1])
        assert information_gain(y, y_left, y_right) == 0.0


class TestGiniImpurity:
    def test_pure_class(self) -> None:
        y = np.array([0, 0, 0, 0])
        assert gini_impurity(y) == 0.0

    def test_binary_balanced(self) -> None:
        y = np.array([0, 0, 1, 1])
        np.testing.assert_allclose(gini_impurity(y), 0.5)

    def test_empty(self) -> None:
        y = np.array([])
        assert gini_impurity(y) == 0.0


class TestGiniGain:
    def test_perfect_split(self) -> None:
        y = np.array([0, 0, 1, 1])
        y_left = np.array([0, 0])
        y_right = np.array([1, 1])
        np.testing.assert_allclose(gini_gain(y, y_left, y_right), 0.5)


class TestMSEReduction:
    def test_perfect_split(self) -> None:
        y = np.array([0.0, 0.0, 10.0, 10.0])
        y_left = np.array([0.0, 0.0])
        y_right = np.array([10.0, 10.0])
        result = mse_reduction(y, y_left, y_right)
        assert result > 0

    def test_no_improvement(self) -> None:
        y = np.array([0.0, 10.0, 0.0, 10.0])
        y_left = np.array([0.0, 10.0])
        y_right = np.array([0.0, 10.0])
        np.testing.assert_allclose(mse_reduction(y, y_left, y_right), 0.0, atol=1e-5)
