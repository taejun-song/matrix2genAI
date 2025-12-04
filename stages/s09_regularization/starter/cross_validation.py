from __future__ import annotations

import numpy as np


def create_folds(
    n_samples: int, n_folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create K-fold cross-validation splits.

    Args:
        n_samples: Total number of samples
        n_folds: Number of folds (K)

    Returns:
        folds: List of (train_indices, val_indices) tuples

    Example:
        >>> folds = create_folds(10, 5)
        >>> len(folds)
        5
        >>> train_idx, val_idx = folds[0]
        >>> len(val_idx)
        2
    """
    # TODO:
    # indices = np.arange(n_samples)
    # fold_sizes = np.full(n_folds, n_samples // n_folds)
    # fold_sizes[:n_samples % n_folds] += 1
    #
    # folds = []
    # current = 0
    # for size in fold_sizes:
    #     val_idx = indices[current:current + size]
    #     train_idx = np.concatenate([indices[:current], indices[current + size:]])
    #     folds.append((train_idx, val_idx))
    #     current += size
    # return folds
    raise NotImplementedError


def cross_val_score(
    model_fn: callable,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    score_fn: callable = None,
) -> np.ndarray:
    """
    Compute cross-validation scores.

    Args:
        model_fn: Function (X_train, y_train) -> predictor function
        X: Features, shape (n_samples, n_features)
        y: Targets, shape (n_samples,)
        n_folds: Number of CV folds
        score_fn: Function (y_true, y_pred) -> score

    Returns:
        scores: Validation scores for each fold, shape (n_folds,)

    Example:
        >>> def train(X, y):
        ...     mean = np.mean(y)
        ...     return lambda X: np.full(len(X), mean)
        >>> scores = cross_val_score(train, X, y, n_folds=5, score_fn=r2_score)
        >>> len(scores)
        5
    """
    # TODO:
    # folds = create_folds(len(X), n_folds)
    # scores = []
    # for train_idx, val_idx in folds:
    #     X_train, X_val = X[train_idx], X[val_idx]
    #     y_train, y_val = y[train_idx], y[val_idx]
    #     model = model_fn(X_train, y_train)
    #     y_pred = model(X_val)
    #     scores.append(score_fn(y_val, y_pred))
    # return np.array(scores)
    raise NotImplementedError


def stratified_folds(
    y: np.ndarray, n_folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified K-fold splits preserving class proportions.

    Args:
        y: Class labels, shape (n_samples,)
        n_folds: Number of folds

    Returns:
        folds: List of (train_indices, val_indices) tuples

    Example:
        >>> y = np.array([0, 0, 0, 0, 1, 1])
        >>> folds = stratified_folds(y, n_folds=2)
        >>> # Each fold has proportional class distribution
    """
    # TODO:
    # classes = np.unique(y)
    # class_indices = {c: np.where(y == c)[0] for c in classes}
    #
    # fold_indices = [[] for _ in range(n_folds)]
    # for c in classes:
    #     indices = class_indices[c]
    #     class_folds = create_folds(len(indices), n_folds)
    #     for fold_idx, (_, val_idx) in enumerate(class_folds):
    #         fold_indices[fold_idx].extend(indices[val_idx])
    #
    # all_indices = np.arange(len(y))
    # folds = []
    # for val_idx in fold_indices:
    #     val_idx = np.array(val_idx)
    #     train_idx = np.setdiff1d(all_indices, val_idx)
    #     folds.append((train_idx, val_idx))
    # return folds
    raise NotImplementedError


def cross_val_predict(
    model_fn: callable,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> np.ndarray:
    """
    Generate cross-validated predictions for all samples.

    Each sample is predicted by a model trained without it.

    Args:
        model_fn: Function (X_train, y_train) -> predictor function
        X: Features, shape (n_samples, n_features)
        y: Targets, shape (n_samples,)
        n_folds: Number of CV folds

    Returns:
        predictions: CV predictions for all samples, shape (n_samples,)

    Example:
        >>> predictions = cross_val_predict(train_fn, X, y, n_folds=5)
        >>> predictions.shape == y.shape
        True
    """
    # TODO:
    # predictions = np.zeros(len(y))
    # folds = create_folds(len(X), n_folds)
    # for train_idx, val_idx in folds:
    #     model = model_fn(X[train_idx], y[train_idx])
    #     predictions[val_idx] = model(X[val_idx])
    # return predictions
    raise NotImplementedError
