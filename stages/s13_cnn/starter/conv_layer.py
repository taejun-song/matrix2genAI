from __future__ import annotations

import numpy as np

from stages.s13_cnn.starter.conv_utils import col2im, get_output_shape, im2col


def conv2d_forward(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward pass for 2D convolution.

    Args:
        x: Input, shape (batch, H, W, C_in)
        W: Weights, shape (kernel_h, kernel_w, C_in, C_out)
        b: Bias, shape (C_out,)
        stride: Stride for convolution
        padding: Zero-padding to apply

    Returns:
        out: Output, shape (batch, H_out, W_out, C_out)
        col: Cached im2col result for backward pass
    """
    # TODO:
    # batch, h_in, w_in, c_in = x.shape
    # k_h, k_w, _, c_out = W.shape
    #
    # h_out, w_out = get_output_shape((h_in, w_in), (k_h, k_w), stride, padding)
    #
    # col = im2col(x, (k_h, k_w), stride, padding)
    # W_col = W.reshape(-1, c_out)
    #
    # out = col @ W_col + b
    # out = out.reshape(batch, h_out, w_out, c_out)
    #
    # return out, col
    raise NotImplementedError


def conv2d_backward(
    grad_output: np.ndarray,
    col: np.ndarray,
    x_shape: tuple[int, int, int, int],
    W: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass for 2D convolution.

    Args:
        grad_output: Gradient from next layer, shape (batch, H_out, W_out, C_out)
        col: Cached im2col from forward pass
        x_shape: Original input shape
        W: Weight matrix
        stride: Stride used in forward
        padding: Padding used in forward

    Returns:
        grad_x: Gradient w.r.t. input, shape x_shape
        grad_W: Gradient w.r.t. weights, shape W.shape
        grad_b: Gradient w.r.t. bias, shape (C_out,)
    """
    # TODO:
    # batch, h_out, w_out, c_out = grad_output.shape
    # k_h, k_w, c_in, _ = W.shape
    #
    # grad_out_col = grad_output.reshape(-1, c_out)
    # W_col = W.reshape(-1, c_out)
    #
    # grad_W_col = col.T @ grad_out_col
    # grad_W = grad_W_col.reshape(W.shape)
    #
    # grad_b = np.sum(grad_out_col, axis=0)
    #
    # grad_col = grad_out_col @ W_col.T
    # grad_x = col2im(grad_col, x_shape, (k_h, k_w), stride, padding)
    #
    # return grad_x, grad_W, grad_b
    raise NotImplementedError


class Conv2D:
    """2D Convolutional layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (filters)
            kernel_size: Size of convolution kernel
            stride: Convolution stride
            padding: Zero-padding
        """
        # TODO:
        # k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        # self.kernel_size = (k, k) if isinstance(kernel_size, int) else kernel_size
        # self.stride = stride
        # self.padding = padding
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        #
        # fan_in = self.kernel_size[0] * self.kernel_size[1] * in_channels
        # std = np.sqrt(2.0 / fan_in)
        # self.W = np.random.randn(self.kernel_size[0], self.kernel_size[1], in_channels, out_channels) * std
        # self.b = np.zeros(out_channels)
        #
        # self.col = None
        # self.x_shape = None
        # self.grad_W = None
        # self.grad_b = None
        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # TODO:
        # self.x_shape = x.shape
        # out, self.col = conv2d_forward(x, self.W, self.b, self.stride, self.padding)
        # return out
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        # TODO:
        # grad_x, self.grad_W, self.grad_b = conv2d_backward(
        #     grad_output, self.col, self.x_shape, self.W, self.stride, self.padding
        # )
        # return grad_x
        raise NotImplementedError

    def get_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (weights, bias)."""
        return self.W, self.b

    def set_params(self, W: np.ndarray, b: np.ndarray) -> None:
        """Set weights and bias."""
        self.W = W
        self.b = b

    def get_gradients(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (grad_W, grad_b)."""
        return self.grad_W, self.grad_b
