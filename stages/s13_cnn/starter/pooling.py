from __future__ import annotations

import numpy as np

from stages.s13_cnn.starter.conv_utils import get_output_shape


def max_pool2d_forward(
    x: np.ndarray,
    pool_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward pass for 2D max pooling.

    Args:
        x: Input, shape (batch, H, W, C)
        pool_size: Size of pooling window
        stride: Stride (default = pool_size for non-overlapping)

    Returns:
        out: Pooled output, shape (batch, H_out, W_out, C)
        indices: Indices of max values for backward pass
    """
    # TODO:
    # p_h, p_w = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
    # if stride is None:
    #     stride = (p_h, p_w)
    # s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    #
    # batch, h_in, w_in, c_in = x.shape
    # h_out, w_out = get_output_shape((h_in, w_in), (p_h, p_w), (s_h, s_w))
    #
    # out = np.zeros((batch, h_out, w_out, c_in))
    # indices = np.zeros((batch, h_out, w_out, c_in, 2), dtype=int)
    #
    # for i in range(h_out):
    #     for j in range(w_out):
    #         h_start, w_start = i * s_h, j * s_w
    #         h_end, w_end = h_start + p_h, w_start + p_w
    #         pool_region = x[:, h_start:h_end, w_start:w_end, :]
    #         pool_flat = pool_region.reshape(batch, -1, c_in)
    #         max_idx = np.argmax(pool_flat, axis=1)
    #         out[:, i, j, :] = np.take_along_axis(pool_flat, max_idx[:, None, :], axis=1).squeeze(1)
    #         indices[:, i, j, :, 0] = h_start + max_idx // p_w
    #         indices[:, i, j, :, 1] = w_start + max_idx % p_w
    #
    # return out, indices
    raise NotImplementedError


def max_pool2d_backward(
    grad_output: np.ndarray,
    indices: np.ndarray,
    input_shape: tuple[int, int, int, int],
    pool_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Backward pass for 2D max pooling.

    Args:
        grad_output: Gradient from next layer, shape (batch, H_out, W_out, C)
        indices: Indices from forward pass
        input_shape: Original input shape
        pool_size: Pool size used in forward
        stride: Stride used in forward

    Returns:
        grad_input: Gradient w.r.t. input, shape input_shape
    """
    # TODO:
    # batch, h_out, w_out, c_in = grad_output.shape
    # grad_input = np.zeros(input_shape)
    #
    # for i in range(h_out):
    #     for j in range(w_out):
    #         for b in range(batch):
    #             for c in range(c_in):
    #                 h_idx, w_idx = indices[b, i, j, c]
    #                 grad_input[b, h_idx, w_idx, c] += grad_output[b, i, j, c]
    #
    # return grad_input
    raise NotImplementedError


class MaxPool2D:
    """2D Max Pooling layer."""

    def __init__(
        self,
        pool_size: int | tuple[int, int] = 2,
        stride: int | tuple[int, int] | None = None,
    ):
        # TODO:
        # self.pool_size = pool_size
        # self.stride = stride
        # self.indices = None
        # self.input_shape = None
        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # TODO:
        # self.input_shape = x.shape
        # out, self.indices = max_pool2d_forward(x, self.pool_size, self.stride)
        # return out
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        # TODO:
        # return max_pool2d_backward(
        #     grad_output, self.indices, self.input_shape, self.pool_size, self.stride
        # )
        raise NotImplementedError


def avg_pool2d_forward(
    x: np.ndarray,
    pool_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Forward pass for 2D average pooling.

    Args:
        x: Input, shape (batch, H, W, C)
        pool_size: Size of pooling window
        stride: Stride (default = pool_size)

    Returns:
        out: Pooled output, shape (batch, H_out, W_out, C)
    """
    # TODO:
    # p_h, p_w = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
    # if stride is None:
    #     stride = (p_h, p_w)
    # s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    #
    # batch, h_in, w_in, c_in = x.shape
    # h_out, w_out = get_output_shape((h_in, w_in), (p_h, p_w), (s_h, s_w))
    #
    # out = np.zeros((batch, h_out, w_out, c_in))
    #
    # for i in range(h_out):
    #     for j in range(w_out):
    #         h_start, w_start = i * s_h, j * s_w
    #         h_end, w_end = h_start + p_h, w_start + p_w
    #         out[:, i, j, :] = np.mean(x[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
    #
    # return out
    raise NotImplementedError


def avg_pool2d_backward(
    grad_output: np.ndarray,
    input_shape: tuple[int, int, int, int],
    pool_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Backward pass for 2D average pooling.

    Args:
        grad_output: Gradient from next layer
        input_shape: Original input shape
        pool_size: Pool size used in forward
        stride: Stride used in forward

    Returns:
        grad_input: Gradient w.r.t. input, shape input_shape
    """
    # TODO:
    # p_h, p_w = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
    # if stride is None:
    #     stride = (p_h, p_w)
    # s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    #
    # batch, h_out, w_out, c_in = grad_output.shape
    # grad_input = np.zeros(input_shape)
    # pool_area = p_h * p_w
    #
    # for i in range(h_out):
    #     for j in range(w_out):
    #         h_start, w_start = i * s_h, j * s_w
    #         h_end, w_end = h_start + p_h, w_start + p_w
    #         grad_input[:, h_start:h_end, w_start:w_end, :] += grad_output[:, i:i+1, j:j+1, :] / pool_area
    #
    # return grad_input
    raise NotImplementedError


class AvgPool2D:
    """2D Average Pooling layer."""

    def __init__(
        self,
        pool_size: int | tuple[int, int] = 2,
        stride: int | tuple[int, int] | None = None,
    ):
        # TODO:
        # self.pool_size = pool_size
        # self.stride = stride
        # self.input_shape = None
        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # TODO:
        # self.input_shape = x.shape
        # return avg_pool2d_forward(x, self.pool_size, self.stride)
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        # TODO:
        # return avg_pool2d_backward(
        #     grad_output, self.input_shape, self.pool_size, self.stride
        # )
        raise NotImplementedError
