from __future__ import annotations

import numpy as np

from stages.s09_regularization.starter.model_selection import (
    best_alpha_ridge,
    grid_search,
    learning_curve,
    validation_curve,
)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1 - ss_res / ss_tot


class TestGridSearch:
    def test_finds_best_param(self) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 2)
        true_w = np.array([1.0, 2.0])
        y = X @ true_w + np.random.randn(50) * 0.1

        def train_scaled(X_train, y_train, scale=1.0):
            w = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            return lambda X: X @ (w * scale)

        param_grid = {'scale': [0.5, 1.0, 1.5]}
        results = grid_search(train_scaled, param_grid, X, y, n_folds=3, score_fn=r2_score)

        assert results['best_params']['scale'] == 1.0
        assert results['best_score'] > 0.9

    def test_multiple_params(self) -> None:
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = np.random.randn(30)

        def train_dummy(X_train, y_train, a=1.0, b=1.0):
            mean = np.mean(y_train) * a * b
            return lambda X: np.full(len(X), mean)

        param_grid = {'a': [0.5, 1.0], 'b': [0.5, 1.0]}
        results = grid_search(train_dummy, param_grid, X, y, n_folds=2, score_fn=r2_score)

        assert 'best_params' in results
        assert 'best_score' in results
        assert 'all_results' in results
        assert len(results['all_results']) == 4

    def test_returns_all_results(self) -> None:
        X = np.random.randn(20, 2)
        y = np.random.randn(20)

        def train_mean(X_train, y_train, p=1.0):
            return lambda X: np.full(len(X), np.mean(y_train))

        param_grid = {'p': [1, 2, 3]}
        results = grid_search(train_mean, param_grid, X, y, n_folds=2, score_fn=r2_score)

        assert len(results['all_results']) == 3
        for r in results['all_results']:
            assert 'params' in r
            assert 'mean_score' in r
            assert 'scores' in r


class TestLearningCurve:
    def test_output_shapes(self) -> None:
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.random.randn(100)

        def train_mean(X_train, y_train):
            return lambda X: np.full(len(X), np.mean(y_train))

        train_sizes = [20, 40, 60, 80]
        sizes, train_scores, val_scores = learning_curve(
            train_mean, X, y, train_sizes, n_folds=3, score_fn=r2_score
        )

        assert len(sizes) == 4
        assert len(train_scores) == 4
        assert len(val_scores) == 4

    def test_scores_improve_with_more_data(self) -> None:
        np.random.seed(42)
        X = np.random.randn(200, 2)
        true_w = np.array([1.0, 2.0])
        y = X @ true_w + np.random.randn(200) * 0.5

        def train_ols(X_train, y_train):
            w = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            return lambda X: X @ w

        train_sizes = [20, 50, 100, 150]
        sizes, train_scores, val_scores = learning_curve(
            train_ols, X, y, train_sizes, n_folds=3, score_fn=r2_score
        )

        assert val_scores[-1] >= val_scores[0] - 0.1

    def test_fraction_sizes(self) -> None:
        X = np.random.randn(100, 2)
        y = np.random.randn(100)

        def train_mean(X_train, y_train):
            return lambda X: np.full(len(X), np.mean(y_train))

        train_sizes = [0.2, 0.5, 0.8]
        sizes, _, _ = learning_curve(
            train_mean, X, y, train_sizes, n_folds=2, score_fn=r2_score
        )

        np.testing.assert_array_equal(sizes, [20, 50, 80])


class TestValidationCurve:
    def test_output_shapes(self) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        def train_scaled(X_train, y_train, scale=1.0):
            w = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            return lambda X: X @ (w * scale)

        param_range = [0.5, 1.0, 1.5, 2.0]
        params, train_scores, val_scores = validation_curve(
            train_scaled, X, y, 'scale', param_range, n_folds=3, score_fn=r2_score
        )

        assert len(params) == 4
        assert len(train_scores) == 4
        assert len(val_scores) == 4

    def test_optimal_param_has_best_val_score(self) -> None:
        np.random.seed(42)
        X = np.random.randn(100, 2)
        true_w = np.array([1.0, 2.0])
        y = X @ true_w + np.random.randn(100) * 0.1

        def train_scaled(X_train, y_train, scale=1.0):
            w = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            return lambda X: X @ (w * scale)

        param_range = [0.5, 0.8, 1.0, 1.2, 1.5]
        params, train_scores, val_scores = validation_curve(
            train_scaled, X, y, 'scale', param_range, n_folds=5, score_fn=r2_score
        )

        best_idx = np.argmax(val_scores)
        assert params[best_idx] == 1.0


class TestBestAlphaRidge:
    def test_finds_reasonable_alpha(self) -> None:
        np.random.seed(42)
        X = np.random.randn(100, 5)
        true_w = np.array([1.0, 2.0, 0.0, 0.0, 0.0])
        y = X @ true_w + np.random.randn(100) * 0.1

        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
        best = best_alpha_ridge(X, y, alphas, n_folds=5, score_fn=r2_score)

        assert best in alphas

    def test_returns_float(self) -> None:
        X = np.random.randn(30, 2)
        y = np.random.randn(30)

        alphas = [0.1, 1.0, 10.0]
        best = best_alpha_ridge(X, y, alphas, n_folds=3, score_fn=r2_score)

        assert isinstance(best, (int, float))

    def test_high_regularization_for_noisy_data(self) -> None:
        np.random.seed(42)
        X = np.random.randn(30, 10)
        y = np.random.randn(30) * 10

        alphas = [0.001, 0.1, 1.0, 10.0, 100.0]
        best = best_alpha_ridge(X, y, alphas, n_folds=3, score_fn=r2_score)

        assert best >= 0.1
