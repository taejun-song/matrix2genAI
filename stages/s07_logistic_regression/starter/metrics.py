from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)

    Returns:
        accuracy: Fraction of correct predictions
    """
    raise NotImplementedError


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        n_classes: Number of classes (auto-detect if None)

    Returns:
        matrix: Confusion matrix, shape (n_classes, n_classes)
    """
    raise NotImplementedError


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary"
) -> tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        average: 'binary' or 'macro'

    Returns:
        precision: Precision score
        recall: Recall score
        f1: F1 score
    """
    raise NotImplementedError


def roc_auc_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute ROC AUC score.

    Args:
        y_true: True binary labels {0, 1}, shape (n_samples,)
        y_pred_proba: Predicted probabilities, shape (n_samples,)

    Returns:
        auc: Area under ROC curve
    """
    raise NotImplementedError


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray | None = None
) -> dict:
    """
    Generate complete classification report.

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        y_pred_proba: Predicted probabilities (optional)

    Returns:
        report: Dictionary with all metrics
    """
    raise NotImplementedError
