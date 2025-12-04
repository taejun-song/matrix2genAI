from __future__ import annotations

import numpy as np

from stages.s09_regularization.starter.cross_validation import (
    create_folds,
    cross_val_predict,
    cross_val_score,
    stratified_folds,
)


class TestCreateFolds:
    def test_basic_5_fold(self) -> None:
        folds = create_folds(n_samples=10, n_folds=5)
        assert len(folds) == 5
        for train_idx, val_idx in folds:
            assert len(val_idx) == 2
            assert len(train_idx) == 8

    def test_all_samples_covered(self) -> None:
        folds = create_folds(n_samples=10, n_folds=5)
        all_val = np.concatenate([val_idx for _, val_idx in folds])
        np.testing.assert_array_equal(np.sort(all_val), np.arange(10))

    def test_no_overlap(self) -> None:
        folds = create_folds(n_samples=10, n_folds=5)
        for train_idx, val_idx in folds:
            assert len(np.intersect1d(train_idx, val_idx)) == 0

    def test_non_divisible(self) -> None:
        folds = create_folds(n_samples=13, n_folds=5)
        val_sizes = [len(val_idx) for _, val_idx in folds]
        assert sum(val_sizes) == 13
        assert max(val_sizes) - min(val_sizes) <= 1

    def test_2_fold(self) -> None:
        folds = create_folds(n_samples=10, n_folds=2)
        assert len(folds) == 2
        assert len(folds[0][1]) == 5
        assert len(folds[1][1]) == 5

    def test_leave_one_out(self) -> None:
        folds = create_folds(n_samples=5, n_folds=5)
        for train_idx, val_idx in folds:
            assert len(val_idx) == 1
            assert len(train_idx) == 4


class TestCrossValScore:
    def test_basic_scoring(self) -> None:
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(100) * 0.1

        def train_mean(X_train, y_train):
            return lambda X: np.full(len(X), np.mean(y_train))

        def mse(y_true, y_pred):
            return -np.mean((y_true - y_pred) ** 2)

        scores = cross_val_score(train_mean, X, y, n_folds=5, score_fn=mse)
        assert len(scores) == 5
        assert all(s < 0 for s in scores)

    def test_perfect_predictor(self) -> None:
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        def train_perfect(X_train, y_train):
            return lambda X: X.ravel()

        def r2(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot == 0:
                return 1.0
            return 1 - ss_res / ss_tot

        scores = cross_val_score(train_perfect, X, y, n_folds=5, score_fn=r2)
        np.testing.assert_allclose(scores, np.ones(5))

    def test_different_folds(self) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        def train_mean(X_train, y_train):
            return lambda X: np.full(len(X), np.mean(y_train))

        def mse(y_true, y_pred):
            return -np.mean((y_true - y_pred) ** 2)

        scores_3 = cross_val_score(train_mean, X, y, n_folds=3, score_fn=mse)
        scores_5 = cross_val_score(train_mean, X, y, n_folds=5, score_fn=mse)
        assert len(scores_3) == 3
        assert len(scores_5) == 5


class TestStratifiedFolds:
    def test_preserves_proportions(self) -> None:
        y = np.array([0] * 80 + [1] * 20)
        folds = stratified_folds(y, n_folds=5)

        for train_idx, val_idx in folds:
            val_ratio = np.mean(y[val_idx])
            assert 0.15 <= val_ratio <= 0.25

    def test_all_samples_covered(self) -> None:
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        folds = stratified_folds(y, n_folds=2)

        all_val = np.concatenate([val_idx for _, val_idx in folds])
        np.testing.assert_array_equal(np.sort(all_val), np.arange(len(y)))

    def test_binary_classification(self) -> None:
        y = np.array([0, 0, 0, 0, 1, 1])
        folds = stratified_folds(y, n_folds=2)

        for train_idx, val_idx in folds:
            assert 0 in y[val_idx]
            assert 1 in y[val_idx] or len(val_idx) < 2

    def test_multiclass(self) -> None:
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        folds = stratified_folds(y, n_folds=3)

        assert len(folds) == 3
        all_val = np.concatenate([val_idx for _, val_idx in folds])
        assert len(all_val) == len(y)


class TestCrossValPredict:
    def test_output_shape(self) -> None:
        X = np.random.randn(20, 3)
        y = np.random.randn(20)

        def train_mean(X_train, y_train):
            return lambda X: np.full(len(X), np.mean(y_train))

        predictions = cross_val_predict(train_mean, X, y, n_folds=5)
        assert predictions.shape == y.shape

    def test_all_samples_predicted(self) -> None:
        X = np.arange(10).reshape(-1, 1)
        y = np.arange(10).astype(float)

        def train_identity(X_train, y_train):
            return lambda X: X.ravel()

        predictions = cross_val_predict(train_identity, X, y, n_folds=5)
        np.testing.assert_allclose(predictions, y)

    def test_no_data_leakage(self) -> None:
        X = np.array([[i] for i in range(10)])
        y = np.arange(10).astype(float)

        def train_memorize(X_train, y_train):
            lookup = dict(zip(X_train.ravel(), y_train))
            return lambda X: np.array([lookup.get(x[0], -999) for x in X])

        predictions = cross_val_predict(train_memorize, X, y, n_folds=5)
        assert np.all(predictions == -999)
