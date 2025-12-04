from __future__ import annotations

import numpy as np

from stages.s15_attention.starter.attention_core import scaled_dot_product_attention


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Split last dimension into multiple heads.

    Args:
        x: Input, shape (batch, seq, d_model)
        num_heads: Number of attention heads

    Returns:
        split: Shape (batch, num_heads, seq, d_k)
    """
    # TODO:
    # batch, seq, d_model = x.shape
    # d_k = d_model // num_heads
    # x = x.reshape(batch, seq, num_heads, d_k)
    # return x.transpose(0, 2, 1, 3)
    raise NotImplementedError


def merge_heads(x: np.ndarray) -> np.ndarray:
    """
    Merge heads back into single dimension.

    Args:
        x: Input, shape (batch, num_heads, seq, d_k)

    Returns:
        merged: Shape (batch, seq, d_model)
    """
    # TODO:
    # batch, num_heads, seq, d_k = x.shape
    # x = x.transpose(0, 2, 1, 3)
    # return x.reshape(batch, seq, num_heads * d_k)
    raise NotImplementedError


def multi_head_attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    W_O: np.ndarray,
    num_heads: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Multi-head attention forward pass.

    Args:
        Q: Queries, shape (batch, seq_q, d_model)
        K: Keys, shape (batch, seq_k, d_model)
        V: Values, shape (batch, seq_k, d_model)
        W_Q, W_K, W_V: Projection matrices
        W_O: Output projection
        num_heads: Number of attention heads
        mask: Optional attention mask

    Returns:
        output: Shape (batch, seq_q, d_model)
        cache: Values for backward pass
    """
    # TODO:
    # Q_proj = Q @ W_Q
    # K_proj = K @ W_K
    # V_proj = V @ W_V
    #
    # Q_heads = split_heads(Q_proj, num_heads)
    # K_heads = split_heads(K_proj, num_heads)
    # V_heads = split_heads(V_proj, num_heads)
    #
    # attn_out, weights = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)
    #
    # concat = merge_heads(attn_out)
    # output = concat @ W_O
    #
    # cache = {
    #     'Q': Q, 'K': K, 'V': V,
    #     'Q_proj': Q_proj, 'K_proj': K_proj, 'V_proj': V_proj,
    #     'Q_heads': Q_heads, 'K_heads': K_heads, 'V_heads': V_heads,
    #     'attn_out': attn_out, 'weights': weights, 'concat': concat,
    #     'W_Q': W_Q, 'W_K': W_K, 'W_V': W_V, 'W_O': W_O,
    #     'num_heads': num_heads
    # }
    #
    # return output, cache
    raise NotImplementedError


def multi_head_attention_backward(
    grad_output: np.ndarray,
    cache: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Multi-head attention backward pass.

    Args:
        grad_output: Gradient from next layer
        cache: Cache from forward pass

    Returns:
        grad_Q, grad_K, grad_V: Gradients w.r.t. inputs
        grad_params: Dict with gradients for W_Q, W_K, W_V, W_O
    """
    # TODO:
    # Q, K, V = cache['Q'], cache['K'], cache['V']
    # Q_heads, K_heads, V_heads = cache['Q_heads'], cache['K_heads'], cache['V_heads']
    # weights = cache['weights']
    # concat = cache['concat']
    # W_Q, W_K, W_V, W_O = cache['W_Q'], cache['W_K'], cache['W_V'], cache['W_O']
    # num_heads = cache['num_heads']
    #
    # grad_W_O = concat.reshape(-1, concat.shape[-1]).T @ grad_output.reshape(-1, grad_output.shape[-1])
    # grad_concat = grad_output @ W_O.T
    #
    # batch, seq_q, d_model = Q.shape
    # d_k = d_model // num_heads
    #
    # grad_attn_out = split_heads(grad_concat, num_heads)
    #
    # grad_V_heads = weights.swapaxes(-2, -1) @ grad_attn_out
    # grad_weights = grad_attn_out @ V_heads.swapaxes(-2, -1)
    #
    # d_softmax = weights * (grad_weights - (grad_weights * weights).sum(axis=-1, keepdims=True))
    # grad_scores = d_softmax / np.sqrt(d_k)
    #
    # grad_Q_heads = grad_scores @ K_heads
    # grad_K_heads = grad_scores.swapaxes(-2, -1) @ Q_heads
    #
    # grad_Q_proj = merge_heads(grad_Q_heads)
    # grad_K_proj = merge_heads(grad_K_heads)
    # grad_V_proj = merge_heads(grad_V_heads)
    #
    # grad_W_Q = Q.reshape(-1, d_model).T @ grad_Q_proj.reshape(-1, d_model)
    # grad_W_K = K.reshape(-1, d_model).T @ grad_K_proj.reshape(-1, d_model)
    # grad_W_V = V.reshape(-1, d_model).T @ grad_V_proj.reshape(-1, d_model)
    #
    # grad_Q = grad_Q_proj @ W_Q.T
    # grad_K = grad_K_proj @ W_K.T
    # grad_V = grad_V_proj @ W_V.T
    #
    # grad_params = {
    #     'W_Q': grad_W_Q, 'W_K': grad_W_K, 'W_V': grad_W_V, 'W_O': grad_W_O
    # }
    #
    # return grad_Q, grad_K, grad_V, grad_params
    raise NotImplementedError


class MultiHeadAttention:
    """Multi-head attention layer."""

    def __init__(self, d_model: int, num_heads: int):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        # TODO:
        # assert d_model % num_heads == 0
        # self.d_model = d_model
        # self.num_heads = num_heads
        # self.d_k = d_model // num_heads
        #
        # scale = np.sqrt(2.0 / d_model)
        # self.W_Q = np.random.randn(d_model, d_model) * scale
        # self.W_K = np.random.randn(d_model, d_model) * scale
        # self.W_V = np.random.randn(d_model, d_model) * scale
        # self.W_O = np.random.randn(d_model, d_model) * scale
        #
        # self.cache = None
        # self.grad_params = None
        raise NotImplementedError

    def forward(
        self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute multi-head attention."""
        # TODO:
        # output, self.cache = multi_head_attention_forward(
        #     Q, K, V, self.W_Q, self.W_K, self.W_V, self.W_O, self.num_heads, mask
        # )
        # return output
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradients."""
        # TODO:
        # grad_Q, grad_K, grad_V, self.grad_params = multi_head_attention_backward(
        #     grad_output, self.cache
        # )
        # return grad_Q, grad_K, grad_V
        raise NotImplementedError

    def get_params(self) -> dict:
        """Return parameters."""
        return {'W_Q': self.W_Q, 'W_K': self.W_K, 'W_V': self.W_V, 'W_O': self.W_O}

    def set_params(self, params: dict) -> None:
        """Set parameters."""
        self.W_Q = params['W_Q']
        self.W_K = params['W_K']
        self.W_V = params['W_V']
        self.W_O = params['W_O']

    def get_gradients(self) -> dict:
        """Return gradients."""
        return self.grad_params
