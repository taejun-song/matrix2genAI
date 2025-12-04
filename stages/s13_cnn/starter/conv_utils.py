from __future__ import annotations

import numpy as np


def pad2d(
    x: np.ndarray,
    padding: int | tuple[int, int],
    value: float = 0.0,
) -> np.ndarray:
    """
    Pad a 4D array (NHWC format).

    Args:
        x: Input array, shape (batch, height, width, channels)
        padding: Pad amount. If int, pad equally. If tuple, (pad_h, pad_w).
        value: Padding value (default 0)

    Returns:
        padded: Shape (batch, height + 2*pad_h, width + 2*pad_w, channels)
    """
    # TODO:
    # if isinstance(padding, int):
    #     pad_h, pad_w = padding, padding
    # else:
    #     pad_h, pad_w = padding
    # if pad_h == 0 and pad_w == 0:
    #     return x
    # return np.pad(
    #     x,
    #     ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
    #     mode='constant',
    #     constant_values=value
    # )
    raise NotImplementedError


def get_output_shape(
    input_shape: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> tuple[int, int]:
    """
    Calculate output shape after convolution/pooling.

    Args:
        input_shape: (height, width)
        kernel_size: Filter size
        stride: Stride
        padding: Padding

    Returns:
        output_shape: (out_height, out_width)
    """
    # TODO:
    # h_in, w_in = input_shape
    # k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    # s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    # p_h, p_w = (padding, padding) if isinstance(padding, int) else padding
    # h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    # w_out = (w_in + 2 * p_w - k_w) // s_w + 1
    # return (h_out, w_out)
    raise NotImplementedError


def im2col(
    x: np.ndarray,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> np.ndarray:
    """
    Transform image to column matrix for efficient convolution.

    Args:
        x: Input, shape (batch, height, width, channels)
        kernel_size: Size of patches to extract
        stride: Stride between patches
        padding: Zero-padding to apply

    Returns:
        col: Shape (batch * out_h * out_w, kernel_h * kernel_w * channels)
    """
    # TODO:
    # x_padded = pad2d(x, padding)
    # batch, h_in, w_in, c_in = x_padded.shape
    # k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    # s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    #
    # h_out, w_out = get_output_shape((h_in, w_in), (k_h, k_w), (s_h, s_w), padding=0)
    #
    # col = np.zeros((batch, h_out, w_out, k_h, k_w, c_in))
    # for i in range(k_h):
    #     i_max = i + s_h * h_out
    #     for j in range(k_w):
    #         j_max = j + s_w * w_out
    #         col[:, :, :, i, j, :] = x_padded[:, i:i_max:s_h, j:j_max:s_w, :]
    #
    # col = col.reshape(batch * h_out * w_out, k_h * k_w * c_in)
    # return col
    raise NotImplementedError


def col2im(
    col: np.ndarray,
    input_shape: tuple[int, int, int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> np.ndarray:
    """
    Transform column matrix back to image format.

    Args:
        col: Column matrix, shape (batch * out_h * out_w, kernel_h * kernel_w * channels)
        input_shape: Original input shape (batch, height, width, channels)
        kernel_size: Kernel size used in im2col
        stride: Stride used in im2col
        padding: Padding used in im2col

    Returns:
        x: Reconstructed input, shape (batch, height, width, channels)
    """
    # TODO:
    # batch, h_in, w_in, c_in = input_shape
    # k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    # s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    # p_h, p_w = (padding, padding) if isinstance(padding, int) else padding
    #
    # h_padded = h_in + 2 * p_h
    # w_padded = w_in + 2 * p_w
    # h_out, w_out = get_output_shape((h_padded, w_padded), (k_h, k_w), (s_h, s_w), padding=0)
    #
    # col_reshaped = col.reshape(batch, h_out, w_out, k_h, k_w, c_in)
    # x_padded = np.zeros((batch, h_padded, w_padded, c_in))
    #
    # for i in range(k_h):
    #     i_max = i + s_h * h_out
    #     for j in range(k_w):
    #         j_max = j + s_w * w_out
    #         x_padded[:, i:i_max:s_h, j:j_max:s_w, :] += col_reshaped[:, :, :, i, j, :]
    #
    # if p_h == 0 and p_w == 0:
    #     return x_padded
    # return x_padded[:, p_h:-p_h, p_w:-p_w, :]
    raise NotImplementedError
