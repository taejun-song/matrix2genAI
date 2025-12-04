from __future__ import annotations

import numpy as np

from stages.s12_feedforward_networks.starter.layer import DenseLayer
from stages.s12_feedforward_networks.starter.neural_network import (
    NeuralNetwork,
    compute_loss,
    compute_loss_gradient,
)
from stages.s12_feedforward_networks.starter.training import create_batches, train


class TestComputeLoss:
    def test_mse_perfect(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        loss = compute_loss(y_true, y_pred, 'mse')
        assert loss == 0.0

    def test_mse_nonzero(self) -> None:
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        loss = compute_loss(y_true, y_pred, 'mse')
        np.testing.assert_allclose(loss, 1.0)


class TestDenseLayer:
    def test_forward_shape(self) -> None:
        layer = DenseLayer(10, 5)
        x = np.random.randn(32, 10)
        out = layer.forward(x)
        assert out.shape == (32, 5)

    def test_backward_shape(self) -> None:
        layer = DenseLayer(10, 5)
        x = np.random.randn(32, 10)
        layer.forward(x)
        grad = np.random.randn(32, 5)
        grad_input = layer.backward(grad)
        assert grad_input.shape == (32, 10)

    def test_gradient_shapes(self) -> None:
        layer = DenseLayer(10, 5)
        x = np.random.randn(32, 10)
        layer.forward(x)
        grad = np.random.randn(32, 5)
        layer.backward(grad)
        grad_W, grad_b = layer.get_gradients()
        assert grad_W.shape == (10, 5)
        assert grad_b.shape == (5,)


class TestNeuralNetwork:
    def test_forward(self) -> None:
        net = NeuralNetwork()
        net.add_layer(DenseLayer(4, 8, activation='relu'))
        net.add_layer(DenseLayer(8, 2))
        x = np.random.randn(10, 4)
        out = net.forward(x)
        assert out.shape == (10, 2)

    def test_train_step_reduces_loss(self) -> None:
        np.random.seed(42)
        net = NeuralNetwork()
        net.add_layer(DenseLayer(2, 4, activation='relu'))
        net.add_layer(DenseLayer(4, 1))

        x = np.random.randn(10, 2)
        y = np.random.randn(10, 1)

        loss1 = net.train_step(x, y, 'mse', learning_rate=0.1)
        loss2 = net.train_step(x, y, 'mse', learning_rate=0.1)
        assert loss2 < loss1


class TestCreateBatches:
    def test_correct_number(self) -> None:
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        batches = create_batches(X, y, batch_size=32, shuffle=False)
        assert len(batches) == 4

    def test_batch_sizes(self) -> None:
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        batches = create_batches(X, y, batch_size=32, shuffle=False)
        sizes = [len(b[0]) for b in batches]
        assert sizes[:3] == [32, 32, 32]
        assert sizes[3] == 4


class TestXORProblem:
    def test_xor_solvable(self) -> None:
        np.random.seed(42)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)

        net = NeuralNetwork()
        net.add_layer(DenseLayer(2, 8, activation='relu'))
        net.add_layer(DenseLayer(8, 1))

        history = train(
            net, X, y, epochs=1000, batch_size=4, learning_rate=0.1, loss_type='mse'
        )

        predictions = net.predict(X)
        rounded = (predictions > 0.5).astype(int)
        accuracy = np.mean(rounded == y)
        assert accuracy >= 0.75
