from __future__ import annotations

import numpy as np

from stages.s11_perceptrons.starter.activations import relu, tanh_activation
from stages.s11_perceptrons.starter.activation_gradients import relu_derivative, tanh_derivative
from stages.s12_feedforward_networks.starter.weight_init import he_normal, zeros


class DenseLayer:
    """Fully connected layer with optional activation."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        activation: str | None = None,
        init: str = "he",
    ):
        """
        Args:
            n_input: Number of input features
            n_output: Number of output features
            activation: 'relu', 'tanh', or None
            init: 'he' or 'xavier'
        """
        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation

        # TODO: Initialize weights and biases
        # self.W = he_normal(n_input, n_output)
        # self.b = zeros((n_output,))

        # Cached values for backward pass
        self.x = None
        self.z = None

        # Gradients
        self.grad_W = None
        self.grad_b = None

        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: z = xW + b, a = activation(z)

        Args:
            x: Input, shape (batch_size, n_input)

        Returns:
            output: Shape (batch_size, n_output)
        """
        # TODO:
        # self.x = x  # Cache for backward
        # self.z = x @ self.W + self.b
        #
        # if self.activation == 'relu':
        #     return relu(self.z)
        # elif self.activation == 'tanh':
        #     return tanh_activation(self.z)
        # return self.z
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients and return gradient for previous layer.

        Args:
            grad_output: Gradient from next layer, shape (batch_size, n_output)

        Returns:
            grad_input: Gradient for previous layer, shape (batch_size, n_input)
        """
        # TODO:
        # batch_size = grad_output.shape[0]
        #
        # # Apply activation derivative
        # if self.activation == 'relu':
        #     dz = grad_output * relu_derivative(self.z)
        # elif self.activation == 'tanh':
        #     dz = grad_output * tanh_derivative(self.z)
        # else:
        #     dz = grad_output
        #
        # # Compute gradients
        # self.grad_W = self.x.T @ dz / batch_size
        # self.grad_b = np.mean(dz, axis=0)
        #
        # # Gradient for previous layer
        # return dz @ self.W.T
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
