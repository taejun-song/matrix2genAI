from __future__ import annotations

import numpy as np

from stages.s15_attention.starter.multi_head import MultiHeadAttention


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Layer normalization.

    Args:
        x: Input, shape (..., d_model)
        gamma: Scale parameter
        beta: Shift parameter
        eps: Numerical stability

    Returns:
        normalized: Same shape as x
    """
    # TODO:
    # mean = x.mean(axis=-1, keepdims=True)
    # var = x.var(axis=-1, keepdims=True)
    # x_norm = (x - mean) / np.sqrt(var + eps)
    # return gamma * x_norm + beta
    raise NotImplementedError


def layer_norm_backward(
    grad_output: np.ndarray,
    x: np.ndarray,
    gamma: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass for layer normalization.

    Args:
        grad_output: Gradient from next layer
        x: Original input
        gamma: Scale parameter
        eps: Epsilon used in forward

    Returns:
        grad_x, grad_gamma, grad_beta
    """
    # TODO:
    # d = x.shape[-1]
    # mean = x.mean(axis=-1, keepdims=True)
    # var = x.var(axis=-1, keepdims=True)
    # std = np.sqrt(var + eps)
    # x_norm = (x - mean) / std
    #
    # grad_beta = grad_output.sum(axis=tuple(range(grad_output.ndim - 1)))
    # grad_gamma = (grad_output * x_norm).sum(axis=tuple(range(grad_output.ndim - 1)))
    #
    # dx_norm = grad_output * gamma
    # dvar = (dx_norm * (x - mean) * -0.5 * (var + eps) ** (-1.5)).sum(axis=-1, keepdims=True)
    # dmean = (dx_norm * -1 / std).sum(axis=-1, keepdims=True) + dvar * (-2 * (x - mean)).mean(axis=-1, keepdims=True)
    # grad_x = dx_norm / std + dvar * 2 * (x - mean) / d + dmean / d
    #
    # return grad_x, grad_gamma, grad_beta
    raise NotImplementedError


def feed_forward(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """
    Position-wise feed-forward network.

    Args:
        x: Input, shape (batch, seq, d_model)
        W1, b1: First layer parameters
        W2, b2: Second layer parameters

    Returns:
        output: Shape (batch, seq, d_model)
    """
    # TODO:
    # hidden = np.maximum(0, x @ W1 + b1)
    # return hidden @ W2 + b2
    raise NotImplementedError


def feed_forward_backward(
    grad_output: np.ndarray,
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass for feed-forward network.

    Returns:
        grad_x, grad_W1, grad_b1, grad_W2, grad_b2
    """
    # TODO:
    # hidden_pre = x @ W1 + b1
    # hidden = np.maximum(0, hidden_pre)
    #
    # grad_hidden = grad_output @ W2.T
    # grad_hidden_pre = grad_hidden * (hidden_pre > 0)
    #
    # batch_seq = x.shape[0] * x.shape[1]
    # x_flat = x.reshape(-1, x.shape[-1])
    # grad_hidden_pre_flat = grad_hidden_pre.reshape(-1, grad_hidden_pre.shape[-1])
    # hidden_flat = hidden.reshape(-1, hidden.shape[-1])
    # grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
    #
    # grad_W2 = hidden_flat.T @ grad_output_flat
    # grad_b2 = grad_output_flat.sum(axis=0)
    # grad_W1 = x_flat.T @ grad_hidden_pre_flat
    # grad_b1 = grad_hidden_pre_flat.sum(axis=0)
    # grad_x = (grad_hidden_pre @ W1.T).reshape(x.shape)
    #
    # return grad_x, grad_W1, grad_b1, grad_W2, grad_b2
    raise NotImplementedError


class TransformerEncoderBlock:
    """Single Transformer encoder block."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: FFN hidden dimension (default: 4 * d_model)
        """
        # TODO:
        # self.d_model = d_model
        # self.num_heads = num_heads
        # self.d_ff = d_ff if d_ff is not None else 4 * d_model
        #
        # self.attention = MultiHeadAttention(d_model, num_heads)
        #
        # self.gamma1 = np.ones(d_model)
        # self.beta1 = np.zeros(d_model)
        # self.gamma2 = np.ones(d_model)
        # self.beta2 = np.zeros(d_model)
        #
        # scale1 = np.sqrt(2.0 / d_model)
        # scale2 = np.sqrt(2.0 / self.d_ff)
        # self.W1 = np.random.randn(d_model, self.d_ff) * scale1
        # self.b1 = np.zeros(self.d_ff)
        # self.W2 = np.random.randn(self.d_ff, d_model) * scale2
        # self.b2 = np.zeros(d_model)
        #
        # self.cache = None
        raise NotImplementedError

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Process input through encoder block."""
        # TODO:
        # x_norm1 = layer_norm(x, self.gamma1, self.beta1)
        # attn_out = self.attention.forward(x_norm1, x_norm1, x_norm1, mask)
        # x = x + attn_out
        #
        # x_norm2 = layer_norm(x, self.gamma2, self.beta2)
        # ff_out = feed_forward(x_norm2, self.W1, self.b1, self.W2, self.b2)
        # x = x + ff_out
        #
        # self.cache = {
        #     'x_input': x,
        #     'x_norm1': x_norm1,
        #     'attn_out': attn_out,
        #     'x_after_attn': x,
        #     'x_norm2': x_norm2,
        #     'ff_out': ff_out
        # }
        #
        # return x
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients."""
        # TODO: Implement backward pass
        # This involves:
        # 1. Backward through FFN and second residual
        # 2. Backward through second layer norm
        # 3. Backward through attention and first residual
        # 4. Backward through first layer norm
        raise NotImplementedError


def stack_encoder_blocks(
    x: np.ndarray,
    blocks: list[TransformerEncoderBlock],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Pass input through stack of encoder blocks.

    Args:
        x: Input embeddings
        blocks: List of encoder blocks
        mask: Optional attention mask

    Returns:
        output: Encoded representations
    """
    # TODO:
    # for block in blocks:
    #     x = block.forward(x, mask)
    # return x
    raise NotImplementedError
