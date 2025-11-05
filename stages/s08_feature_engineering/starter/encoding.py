from __future__ import annotations

import numpy as np


def label_encode(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode categorical labels as integers.

    Args:
        y: Categorical labels, shape (n_samples,)

    Returns:
        y_encoded: Integer labels, shape (n_samples,)
        classes: Unique classes, shape (n_classes,)
    """
    raise NotImplementedError


def label_decode(y_encoded: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    Decode integer labels back to categories.

    Args:
        y_encoded: Integer labels, shape (n_samples,)
        classes: Original class labels, shape (n_classes,)

    Returns:
        y: Categorical labels, shape (n_samples,)
    """
    raise NotImplementedError


def one_hot_encode(y: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """
    Convert integer labels to one-hot encoding.

    Args:
        y: Integer labels, shape (n_samples,)
        n_classes: Number of classes (auto-detect if None)

    Returns:
        y_onehot: One-hot encoded, shape (n_samples, n_classes)
    """
    raise NotImplementedError


def one_hot_decode(y_onehot: np.ndarray) -> np.ndarray:
    """
    Convert one-hot encoding back to integer labels.

    Args:
        y_onehot: One-hot encoded, shape (n_samples, n_classes)

    Returns:
        y: Integer labels, shape (n_samples,)
    """
    raise NotImplementedError
