# Stage 13: Convolutional Neural Networks - Specifications

## Module: conv_utils.py

### pad2d
```python
def pad2d(
    x: np.ndarray,
    padding: int | tuple[int, int],
    value: float = 0.0
) -> np.ndarray:
    """
    Pad a 4D array (NHWC format).

    Args:
        x: Input array, shape (batch, height, width, channels)
        padding: Pad amount. If int, pad equally. If tuple, (pad_h, pad_w).
        value: Padding value (default 0)

    Returns:
        padded: Shape (batch, height + 2*pad_h, width + 2*pad_w, channels)

    Example:
        >>> x = np.ones((1, 3, 3, 1))
        >>> padded = pad2d(x, 1)
        >>> padded.shape
        (1, 5, 5, 1)
    """
```

### get_output_shape
```python
def get_output_shape(
    input_shape: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0
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

    Formula:
        out = floor((input + 2*padding - kernel) / stride) + 1
    """
```

### im2col
```python
def im2col(
    x: np.ndarray,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0
) -> np.ndarray:
    """
    Transform image to column matrix for efficient convolution.

    Each row contains a flattened patch of the input.

    Args:
        x: Input, shape (batch, height, width, channels)
        kernel_size: Size of patches to extract
        stride: Stride between patches
        padding: Zero-padding to apply

    Returns:
        col: Shape (batch * out_h * out_w, kernel_h * kernel_w * channels)

    Example:
        >>> x = np.arange(16).reshape(1, 4, 4, 1)
        >>> col = im2col(x, kernel_size=2, stride=2)
        >>> col.shape
        (4, 4)  # 4 patches of 2*2*1
    """
```

### col2im
```python
def col2im(
    col: np.ndarray,
    input_shape: tuple[int, int, int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0
) -> np.ndarray:
    """
    Transform column matrix back to image format.

    This is the "inverse" of im2col, accumulating overlapping patches.

    Args:
        col: Column matrix, shape (batch * out_h * out_w, kernel_h * kernel_w * channels)
        input_shape: Original input shape (batch, height, width, channels)
        kernel_size: Kernel size used in im2col
        stride: Stride used in im2col
        padding: Padding used in im2col

    Returns:
        x: Reconstructed input, shape (batch, height, width, channels)

    Note:
        For stride=1 and overlapping patches, values are accumulated.
    """
```

---

## Module: conv_layer.py

### conv2d_forward
```python
def conv2d_forward(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    stride: int = 1,
    padding: int = 0
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

    Implementation:
        1. Apply im2col to input
        2. Reshape W to (k*k*C_in, C_out)
        3. Compute col @ W_reshaped + b
        4. Reshape to output dimensions
    """
```

### conv2d_backward
```python
def conv2d_backward(
    grad_output: np.ndarray,
    col: np.ndarray,
    x_shape: tuple[int, int, int, int],
    W: np.ndarray,
    stride: int = 1,
    padding: int = 0
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

    Implementation:
        1. Reshape grad_output to (batch * H_out * W_out, C_out)
        2. grad_W = col.T @ grad_output_reshaped (then reshape)
        3. grad_b = sum over spatial and batch dimensions
        4. grad_col = grad_output_reshaped @ W_reshaped.T
        5. grad_x = col2im(grad_col, ...)
    """
```

### Conv2D (class)
```python
class Conv2D:
    """
    2D Convolutional layer.

    Attributes:
        W: Weights, shape (kernel_h, kernel_w, C_in, C_out)
        b: Bias, shape (C_out,)
        stride: Convolution stride
        padding: Zero-padding amount

    Methods:
        forward(x): Compute convolution
        backward(grad_output): Compute gradients
        get_params(): Return (W, b)
        set_params(W, b): Set parameters
        get_gradients(): Return (grad_W, grad_b)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int = 0
    ):
        """
        Initialize Conv2D layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (filters)
            kernel_size: Size of convolution kernel
            stride: Convolution stride
            padding: Zero-padding

        Weight initialization:
            W ~ He normal with fan_in = kernel_h * kernel_w * in_channels
        """
```

---

## Module: pooling.py

### max_pool2d_forward
```python
def max_pool2d_forward(
    x: np.ndarray,
    pool_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None
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

    Example:
        >>> x = np.array([[[[1, 2], [3, 4]]]]).transpose(0, 2, 3, 1)  # (1,2,2,2)
        >>> out, _ = max_pool2d_forward(x, pool_size=2)
        >>> out.squeeze()  # [3, 4] - max of each channel
    """
```

### max_pool2d_backward
```python
def max_pool2d_backward(
    grad_output: np.ndarray,
    indices: np.ndarray,
    input_shape: tuple[int, int, int, int],
    pool_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None
) -> np.ndarray:
    """
    Backward pass for 2D max pooling.

    Gradient flows only to the position that was the maximum.

    Args:
        grad_output: Gradient from next layer, shape (batch, H_out, W_out, C)
        indices: Indices from forward pass
        input_shape: Original input shape
        pool_size: Pool size used in forward
        stride: Stride used in forward

    Returns:
        grad_input: Gradient w.r.t. input, shape input_shape
    """
```

### MaxPool2D (class)
```python
class MaxPool2D:
    """
    2D Max Pooling layer.

    Attributes:
        pool_size: Pooling window size
        stride: Pooling stride

    Methods:
        forward(x): Apply max pooling
        backward(grad_output): Compute gradient
    """

    def __init__(
        self,
        pool_size: int | tuple[int, int] = 2,
        stride: int | tuple[int, int] | None = None
    ):
        """
        Args:
            pool_size: Size of pooling window
            stride: Stride (default = pool_size)
        """
```

### avg_pool2d_forward
```python
def avg_pool2d_forward(
    x: np.ndarray,
    pool_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None
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
```

### avg_pool2d_backward
```python
def avg_pool2d_backward(
    grad_output: np.ndarray,
    input_shape: tuple[int, int, int, int],
    pool_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None
) -> np.ndarray:
    """
    Backward pass for 2D average pooling.

    Gradient is distributed equally to all positions in each pool.

    Args:
        grad_output: Gradient from next layer
        input_shape: Original input shape
        pool_size: Pool size used in forward
        stride: Stride used in forward

    Returns:
        grad_input: Gradient w.r.t. input, shape input_shape
    """
```

### AvgPool2D (class)
```python
class AvgPool2D:
    """
    2D Average Pooling layer.

    Methods:
        forward(x): Apply average pooling
        backward(grad_output): Compute gradient
    """
```

---

## Module: cnn.py

### Flatten (class)
```python
class Flatten:
    """
    Flatten spatial dimensions for transition to fully connected layers.

    Example:
        >>> flatten = Flatten()
        >>> x = np.random.randn(32, 7, 7, 64)
        >>> out = flatten.forward(x)
        >>> out.shape
        (32, 3136)  # 7 * 7 * 64

    Methods:
        forward(x): Flatten spatial dims, keeping batch
        backward(grad_output): Restore original shape
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Flatten input.

        Args:
            x: Input, shape (batch, H, W, C)

        Returns:
            out: Flattened, shape (batch, H * W * C)
        """

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Reshape gradient back to original shape.

        Args:
            grad_output: Gradient, shape (batch, H * W * C)

        Returns:
            grad_input: Reshaped, shape (batch, H, W, C)
        """
```

### build_lenet
```python
def build_lenet(num_classes: int = 10) -> list:
    """
    Build LeNet-5 style architecture.

    Architecture:
        Conv2D(1, 6, 5) -> ReLU -> MaxPool(2)
        Conv2D(6, 16, 5) -> ReLU -> MaxPool(2)
        Flatten
        Dense(400, 120) -> ReLU
        Dense(120, 84) -> ReLU
        Dense(84, num_classes)

    Args:
        num_classes: Number of output classes

    Returns:
        layers: List of layer objects

    Note:
        Assumes 32x32 grayscale input (like MNIST padded to 32x32).
        For 28x28 input, first conv should use padding=2.
    """
```

### forward_cnn
```python
def forward_cnn(layers: list, x: np.ndarray) -> np.ndarray:
    """
    Forward pass through a list of CNN layers.

    Args:
        layers: List of layer objects with forward() method
        x: Input batch

    Returns:
        output: Network output
    """
```

### backward_cnn
```python
def backward_cnn(layers: list, grad_output: np.ndarray) -> None:
    """
    Backward pass through a list of CNN layers.

    Args:
        layers: List of layer objects with backward() method
        grad_output: Loss gradient

    Note:
        Iterates layers in reverse, passing gradients backward.
    """
```
