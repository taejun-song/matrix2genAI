from __future__ import annotations

from itertools import product

import numpy as np

from stages.s09_regularization.starter.cross_validation import (
    create_folds,
    cross_val_score,
)


def grid_search(
    model_fn: callable,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    score_fn: callable = None,
) -> dict:
    """
    Grid search with cross-validation for hyperparameter tuning.

    Args:
        model_fn: Function (X, y, **params) -> predictor function
        param_grid: Dict mapping param names to lists of values
        X: Features, shape (n_samples, n_features)
        y: Targets, shape (n_samples,)
        n_folds: Number of CV folds
        score_fn: Function (y_true, y_pred) -> score

    Returns:
        results: Dict with 'best_params', 'best_score', 'all_results'

    Example:
        >>> param_grid = {'alpha': [0.01, 0.1, 1.0]}
        >>> results = grid_search(train_ridge, param_grid, X, y)
        >>> results['best_params']
        {'alpha': 0.1}
    """
    # TODO:
    # param_names = list(param_grid.keys())
    # param_values = list(param_grid.values())
    #
    # best_score = -np.inf
    # best_params = None
    # all_results = []
    #
    # for values in product(*param_values):
    #     params = dict(zip(param_names, values))
    #
    #     def make_model(X_tr, y_tr, p=params):
    #         return model_fn(X_tr, y_tr, **p)
    #
    #     scores = cross_val_score(make_model, X, y, n_folds, score_fn)
    #     mean_score = np.mean(scores)
    #
    #     all_results.append({
    #         'params': params,
    #         'mean_score': mean_score,
    #         'scores': scores
    #     })
    #
    #     if mean_score > best_score:
    #         best_score = mean_score
    #         best_params = params
    #
    # return {
    #     'best_params': best_params,
    #     'best_score': best_score,
    #     'all_results': all_results
    # }
    raise NotImplementedError


def learning_curve(
    model_fn: callable,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: list[int | float],
    n_folds: int = 5,
    score_fn: callable = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute learning curve: scores vs training set size.

    Args:
        model_fn: Function (X, y) -> predictor function
        X: Features, shape (n_samples, n_features)
        y: Targets, shape (n_samples,)
        train_sizes: List of sizes (int) or fractions (float in [0,1])
        n_folds: Number of CV folds
        score_fn: Function (y_true, y_pred) -> score

    Returns:
        train_sizes: Actual training sizes used
        train_scores: Mean training scores at each size
        val_scores: Mean validation scores at each size

    Example:
        >>> sizes, train_sc, val_sc = learning_curve(
        ...     train_fn, X, y, train_sizes=[50, 100, 200]
        ... )
        >>> len(train_sc) == len(val_sc) == 3
        True
    """
    # TODO:
    # n_samples = len(X)
    # actual_sizes = []
    # train_scores = []
    # val_scores = []
    #
    # for size in train_sizes:
    #     if isinstance(size, float):
    #         size = int(size * n_samples)
    #     actual_sizes.append(size)
    #
    #     X_subset, y_subset = X[:size], y[:size]
    #     folds = create_folds(size, min(n_folds, size))
    #
    #     fold_train = []
    #     fold_val = []
    #     for train_idx, val_idx in folds:
    #         model = model_fn(X_subset[train_idx], y_subset[train_idx])
    #         fold_train.append(score_fn(y_subset[train_idx], model(X_subset[train_idx])))
    #         fold_val.append(score_fn(y_subset[val_idx], model(X_subset[val_idx])))
    #
    #     train_scores.append(np.mean(fold_train))
    #     val_scores.append(np.mean(fold_val))
    #
    # return np.array(actual_sizes), np.array(train_scores), np.array(val_scores)
    raise NotImplementedError


def validation_curve(
    model_fn: callable,
    X: np.ndarray,
    y: np.ndarray,
    param_name: str,
    param_range: list,
    n_folds: int = 5,
    score_fn: callable = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute validation curve: scores vs hyperparameter value.

    Args:
        model_fn: Function (X, y, **{param_name: value}) -> predictor
        X: Features, shape (n_samples, n_features)
        y: Targets, shape (n_samples,)
        param_name: Name of hyperparameter to vary
        param_range: List of values to try
        n_folds: Number of CV folds
        score_fn: Function (y_true, y_pred) -> score

    Returns:
        param_range: Parameter values
        train_scores: Mean training scores
        val_scores: Mean validation scores

    Example:
        >>> params, train_sc, val_sc = validation_curve(
        ...     train_ridge, X, y, 'alpha', [0.01, 0.1, 1.0]
        ... )
    """
    # TODO:
    # train_scores = []
    # val_scores = []
    #
    # for param_value in param_range:
    #     def make_model(X_tr, y_tr, pv=param_value):
    #         return model_fn(X_tr, y_tr, **{param_name: pv})
    #
    #     folds = create_folds(len(X), n_folds)
    #     fold_train = []
    #     fold_val = []
    #
    #     for train_idx, val_idx in folds:
    #         model = make_model(X[train_idx], y[train_idx])
    #         fold_train.append(score_fn(y[train_idx], model(X[train_idx])))
    #         fold_val.append(score_fn(y[val_idx], model(X[val_idx])))
    #
    #     train_scores.append(np.mean(fold_train))
    #     val_scores.append(np.mean(fold_val))
    #
    # return np.array(param_range), np.array(train_scores), np.array(val_scores)
    raise NotImplementedError


def best_alpha_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alphas: list[float],
    n_folds: int = 5,
    score_fn: callable = None,
) -> float:
    """
    Find optimal Ridge regularization strength via cross-validation.

    Args:
        X: Features, shape (n_samples, n_features)
        y: Targets, shape (n_samples,)
        alphas: List of alpha values to try
        n_folds: Number of CV folds
        score_fn: Function (y_true, y_pred) -> score

    Returns:
        best_alpha: Optimal regularization strength

    Example:
        >>> best = best_alpha_ridge(X, y, alphas=[0.01, 0.1, 1.0, 10.0])
        >>> best in [0.01, 0.1, 1.0, 10.0]
        True
    """
    # TODO:
    # best_score = -np.inf
    # best_alpha = alphas[0]
    #
    # for alpha in alphas:
    #     def train_ridge(X_tr, y_tr, a=alpha):
    #         # Simple ridge: solve (X^T X + alpha*I)^-1 X^T y
    #         n_features = X_tr.shape[1]
    #         XtX = X_tr.T @ X_tr
    #         Xty = X_tr.T @ y_tr
    #         w = np.linalg.solve(XtX + a * np.eye(n_features), Xty)
    #         return lambda X_new: X_new @ w
    #
    #     scores = cross_val_score(train_ridge, X, y, n_folds, score_fn)
    #     mean_score = np.mean(scores)
    #
    #     if mean_score > best_score:
    #         best_score = mean_score
    #         best_alpha = alpha
    #
    # return best_alpha
    raise NotImplementedError
