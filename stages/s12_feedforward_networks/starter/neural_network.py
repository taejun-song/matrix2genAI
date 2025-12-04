from __future__ import annotations

import numpy as np

from stages.s12_feedforward_networks.starter.layer import DenseLayer


def compute_loss(y_true: np.ndarray, y_pred: np.ndarray, loss_type: str) -> float:
    """
    Compute loss value.

    Args:
        y_true: True labels, shape (n_samples,) or (n_samples, n_classes)
        y_pred: Predictions, shape (n_samples,) or (n_samples, n_classes)
        loss_type: 'mse' or 'cross_entropy'

    Returns:
        loss: Scalar loss value
    """
    # TODO:
    # if loss_type == 'mse':
    #     return float(np.mean((y_true - y_pred) ** 2))
    # elif loss_type == 'cross_entropy':
    #     eps = 1e-15
    #     y_pred = np.clip(y_pred, eps, 1 - eps)
    #     return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=-1)))
    raise NotImplementedError


def compute_loss_gradient(
    y_true: np.ndarray, y_pred: np.ndarray, loss_type: str
) -> np.ndarray:
    """
    Compute gradient of loss w.r.t. predictions.

    Args:
        y_true: True labels
        y_pred: Predictions
        loss_type: 'mse' or 'cross_entropy'

    Returns:
        gradient: Same shape as y_pred
    """
    # TODO:
    # n = len(y_true)
    # if loss_type == 'mse':
    #     return (2 / n) * (y_pred - y_true)
    # elif loss_type == 'cross_entropy':
    #     # For softmax + cross-entropy
    #     return (y_pred - y_true) / n
    raise NotImplementedError


class NeuralNetwork:
    """Feedforward neural network."""

    def __init__(self):
        self.layers: list[DenseLayer] = []

    def add_layer(self, layer: DenseLayer) -> None:
        """Add a layer to the network."""
        self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers.

        Args:
            x: Input, shape (batch_size, n_features)

        Returns:
            output: Network output
        """
        # TODO:
        # for layer in self.layers:
        #     x = layer.forward(x)
        # return x
        raise NotImplementedError

    def backward(self, loss_gradient: np.ndarray) -> None:
        """
        Backward pass through all layers.

        Args:
            loss_gradient: Gradient of loss w.r.t. network output
        """
        # TODO:
        # grad = loss_gradient
        # for layer in reversed(self.layers):
        #     grad = layer.backward(grad)
        raise NotImplementedError

    def train_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        loss_type: str,
        learning_rate: float,
    ) -> float:
        """
        Single training step: forward, loss, backward, update.

        Args:
            x: Input batch
            y: Target batch
            loss_type: 'mse' or 'cross_entropy'
            learning_rate: Learning rate

        Returns:
            loss: Loss value for this batch
        """
        # TODO:
        # # Forward
        # y_pred = self.forward(x)
        #
        # # Loss
        # loss = compute_loss(y, y_pred, loss_type)
        #
        # # Backward
        # loss_grad = compute_loss_gradient(y, y_pred, loss_type)
        # self.backward(loss_grad)
        #
        # # Update weights
        # for layer in self.layers:
        #     grad_W, grad_b = layer.get_gradients()
        #     W, b = layer.get_params()
        #     layer.set_params(
        #         W - learning_rate * grad_W,
        #         b - learning_rate * grad_b
        #     )
        #
        # return loss
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict without training."""
        return self.forward(x)
