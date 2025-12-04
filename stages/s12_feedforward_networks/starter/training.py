from __future__ import annotations

import numpy as np

from stages.s12_feedforward_networks.starter.neural_network import NeuralNetwork, compute_loss


def create_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create mini-batches from data.

    Args:
        X: Features, shape (n_samples, n_features)
        y: Labels, shape (n_samples,) or (n_samples, n_classes)
        batch_size: Size of each batch
        shuffle: Whether to shuffle data

    Returns:
        batches: List of (X_batch, y_batch) tuples
    """
    # TODO:
    # n_samples = len(X)
    # indices = np.arange(n_samples)
    # if shuffle:
    #     np.random.shuffle(indices)
    #
    # batches = []
    # for start in range(0, n_samples, batch_size):
    #     end = min(start + batch_size, n_samples)
    #     batch_idx = indices[start:end]
    #     batches.append((X[batch_idx], y[batch_idx]))
    # return batches
    raise NotImplementedError


def train_epoch(
    network: NeuralNetwork,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    learning_rate: float,
    loss_type: str,
) -> float:
    """
    Train for one epoch.

    Args:
        network: Neural network
        X: Training features
        y: Training labels
        batch_size: Batch size
        learning_rate: Learning rate
        loss_type: Loss function type

    Returns:
        avg_loss: Average loss over all batches
    """
    # TODO:
    # batches = create_batches(X, y, batch_size)
    # total_loss = 0.0
    # for X_batch, y_batch in batches:
    #     loss = network.train_step(X_batch, y_batch, loss_type, learning_rate)
    #     total_loss += loss
    # return total_loss / len(batches)
    raise NotImplementedError


def early_stopping(val_losses: list[float], patience: int) -> bool:
    """
    Check if training should stop early.

    Args:
        val_losses: List of validation losses
        patience: Number of epochs to wait for improvement

    Returns:
        should_stop: True if no improvement for 'patience' epochs
    """
    # TODO:
    # if len(val_losses) <= patience:
    #     return False
    # best_loss = min(val_losses[:-patience])
    # recent_best = min(val_losses[-patience:])
    # return recent_best >= best_loss
    raise NotImplementedError


def train(
    network: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    loss_type: str = "mse",
    patience: int | None = None,
) -> dict:
    """
    Train neural network.

    Args:
        network: Neural network
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        loss_type: 'mse' or 'cross_entropy'
        patience: Early stopping patience (None = no early stopping)

    Returns:
        history: Dict with 'train_loss' and 'val_loss' lists
    """
    # TODO:
    # history = {'train_loss': [], 'val_loss': []}
    #
    # for epoch in range(epochs):
    #     train_loss = train_epoch(
    #         network, X_train, y_train, batch_size, learning_rate, loss_type
    #     )
    #     history['train_loss'].append(train_loss)
    #
    #     if X_val is not None:
    #         y_pred = network.predict(X_val)
    #         val_loss = compute_loss(y_val, y_pred, loss_type)
    #         history['val_loss'].append(val_loss)
    #
    #         if patience and early_stopping(history['val_loss'], patience):
    #             break
    #
    # return history
    raise NotImplementedError
