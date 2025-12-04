from __future__ import annotations

import numpy as np

from stages.s12_feedforward_networks.starter.layer import DenseLayer
from stages.s13_cnn.starter.conv_layer import Conv2D
from stages.s13_cnn.starter.pooling import MaxPool2D


class ReLU:
    """ReLU activation layer for CNN."""

    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # TODO:
        # self.mask = (x > 0)
        # return x * self.mask
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        # TODO:
        # return grad_output * self.mask
        raise NotImplementedError


class Flatten:
    """Flatten spatial dimensions for transition to fully connected layers."""

    def __init__(self):
        self.input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Flatten input.

        Args:
            x: Input, shape (batch, H, W, C)

        Returns:
            out: Flattened, shape (batch, H * W * C)
        """
        # TODO:
        # self.input_shape = x.shape
        # return x.reshape(x.shape[0], -1)
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Reshape gradient back to original shape.

        Args:
            grad_output: Gradient, shape (batch, H * W * C)

        Returns:
            grad_input: Reshaped, shape (batch, H, W, C)
        """
        # TODO:
        # return grad_output.reshape(self.input_shape)
        raise NotImplementedError


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
        Assumes 32x32 grayscale input.
        For 28x28 input, use padding=2 in first conv.
    """
    # TODO:
    # layers = [
    #     Conv2D(1, 6, kernel_size=5, padding=0),
    #     ReLU(),
    #     MaxPool2D(pool_size=2),
    #     Conv2D(6, 16, kernel_size=5, padding=0),
    #     ReLU(),
    #     MaxPool2D(pool_size=2),
    #     Flatten(),
    #     DenseLayer(400, 120, activation='relu'),
    #     DenseLayer(120, 84, activation='relu'),
    #     DenseLayer(84, num_classes),
    # ]
    # return layers
    raise NotImplementedError


def forward_cnn(layers: list, x: np.ndarray) -> np.ndarray:
    """
    Forward pass through a list of CNN layers.

    Args:
        layers: List of layer objects with forward() method
        x: Input batch

    Returns:
        output: Network output
    """
    # TODO:
    # for layer in layers:
    #     x = layer.forward(x)
    # return x
    raise NotImplementedError


def backward_cnn(layers: list, grad_output: np.ndarray) -> None:
    """
    Backward pass through a list of CNN layers.

    Args:
        layers: List of layer objects with backward() method
        grad_output: Loss gradient
    """
    # TODO:
    # grad = grad_output
    # for layer in reversed(layers):
    #     grad = layer.backward(grad)
    raise NotImplementedError
