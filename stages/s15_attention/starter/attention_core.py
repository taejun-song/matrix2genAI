from __future__ import annotations

import numpy as np


def compute_attention_scores(
    Q: np.ndarray,
    K: np.ndarray,
    scale: bool = True,
) -> np.ndarray:
    """
    Compute attention scores between queries and keys.

    Args:
        Q: Queries, shape (..., seq_q, d_k)
        K: Keys, shape (..., seq_k, d_k)
        scale: Whether to scale by √d_k

    Returns:
        scores: Shape (..., seq_q, seq_k)
    """
    # TODO:
    # scores = Q @ K.swapaxes(-2, -1)
    # if scale:
    #     d_k = Q.shape[-1]
    #     scores = scores / np.sqrt(d_k)
    # return scores
    raise NotImplementedError


def apply_attention_mask(
    scores: np.ndarray,
    mask: np.ndarray,
    mask_value: float = -1e9,
) -> np.ndarray:
    """
    Apply mask to attention scores.

    Args:
        scores: Attention scores
        mask: Boolean mask (True = attend, False = mask out)
        mask_value: Value for masked positions

    Returns:
        masked_scores: Same shape as scores
    """
    # TODO:
    # return np.where(mask, scores, mask_value)
    raise NotImplementedError


def attention_weights(
    scores: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """
    Convert scores to attention weights via softmax.

    Args:
        scores: Attention scores
        axis: Axis to apply softmax

    Returns:
        weights: Same shape, sums to 1 along axis
    """
    # TODO:
    # scores_max = np.max(scores, axis=axis, keepdims=True)
    # exp_scores = np.exp(scores - scores_max)
    # return exp_scores / np.sum(exp_scores, axis=axis, keepdims=True)
    raise NotImplementedError


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Complete scaled dot-product attention.

    Args:
        Q: Queries, shape (..., seq_q, d_k)
        K: Keys, shape (..., seq_k, d_k)
        V: Values, shape (..., seq_k, d_v)
        mask: Optional mask

    Returns:
        output: Attended values
        weights: Attention weights
    """
    # TODO:
    # scores = compute_attention_scores(Q, K, scale=True)
    # if mask is not None:
    #     scores = apply_attention_mask(scores, mask)
    # weights = attention_weights(scores)
    # output = weights @ V
    # return output, weights
    raise NotImplementedError


def additive_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bahdanau-style additive attention.

    Args:
        Q: Queries, shape (batch, seq_q, d_q)
        K: Keys, shape (batch, seq_k, d_k)
        V: Values, shape (batch, seq_k, d_v)
        W_q: Query projection
        W_k: Key projection
        v: Score vector

    Returns:
        output: Attended values
        weights: Attention weights
    """
    # TODO:
    # batch, seq_q, _ = Q.shape
    # _, seq_k, _ = K.shape
    #
    # Q_proj = Q @ W_q
    # K_proj = K @ W_k
    #
    # Q_exp = Q_proj[:, :, np.newaxis, :]
    # K_exp = K_proj[:, np.newaxis, :, :]
    #
    # scores = np.tanh(Q_exp + K_exp) @ v
    #
    # if mask is not None:
    #     scores = apply_attention_mask(scores, mask)
    #
    # weights = attention_weights(scores, axis=-1)
    # output = weights @ V
    #
    # return output, weights
    raise NotImplementedError
