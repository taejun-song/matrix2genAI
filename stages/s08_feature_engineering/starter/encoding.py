from __future__ import annotations

import numpy as np

# ============================================================================
# Building Block 1: Label Encoding
# ============================================================================


def _find_unique_classes(y: np.ndarray) -> np.ndarray:
    """
    Find unique classes in sorted order.

    Args:
        y: Categorical labels

    Returns:
        Unique classes, sorted
    """
    # TODO: return np.unique(y)
    raise NotImplementedError


def label_encode(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode categorical → integers: ['cat', 'dog', 'cat'] → [0, 1, 0]

    Args:
        y: Categorical labels

    Returns:
        y_encoded: Integer labels
        classes: Unique classes
    """
    # TODO:
    # classes = _find_unique_classes(y)
    # y_encoded = np.searchsorted(classes, y)
    # return y_encoded, classes
    raise NotImplementedError


def label_decode(y_encoded: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    Decode integers → categorical: [0, 1, 0] → ['cat', 'dog', 'cat']

    Args:
        y_encoded: Integer labels
        classes: Original class labels

    Returns:
        y: Categorical labels
    """
    # TODO: return classes[y_encoded]
    raise NotImplementedError


# ============================================================================
# Building Block 2: One-Hot Encoding
# ============================================================================


def one_hot_encode(y: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """
    Convert integers → one-hot: [0, 1, 0] → [[1,0], [0,1], [1,0]]

    Args:
        y: Integer labels
        n_classes: Number of classes (auto-detect if None)

    Returns:
        y_onehot: One-hot encoded
    """
    # TODO:
    # if n_classes is None:
    #     n_classes = int(np.max(y)) + 1
    # return np.eye(n_classes)[y]
    raise NotImplementedError


def one_hot_decode(y_onehot: np.ndarray) -> np.ndarray:
    """
    Convert one-hot → integers: [[1,0], [0,1], [1,0]] → [0, 1, 0]

    Args:
        y_onehot: One-hot encoded

    Returns:
        y: Integer labels
    """
    # TODO: return np.argmax(y_onehot, axis=1)
    raise NotImplementedError
