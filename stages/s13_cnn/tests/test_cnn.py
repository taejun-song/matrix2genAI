from __future__ import annotations

import numpy as np

from stages.s13_cnn.starter.cnn import (
    Flatten,
    ReLU,
    backward_cnn,
    build_lenet,
    forward_cnn,
)
from stages.s13_cnn.starter.conv_layer import Conv2D
from stages.s13_cnn.starter.pooling import MaxPool2D


class TestReLU:
    def test_forward_positive(self) -> None:
        relu = ReLU()
        x = np.array([1.0, 2.0, 3.0])
        out = relu.forward(x)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_forward_negative(self) -> None:
        relu = ReLU()
        x = np.array([-1.0, -2.0, 0.0])
        out = relu.forward(x)
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])

    def test_backward(self) -> None:
        relu = ReLU()
        x = np.array([-1.0, 2.0, -3.0, 4.0])
        relu.forward(x)
        grad_out = np.ones(4)
        grad_x = relu.backward(grad_out)
        np.testing.assert_array_equal(grad_x, [0.0, 1.0, 0.0, 1.0])


class TestFlatten:
    def test_forward_shape(self) -> None:
        flatten = Flatten()
        x = np.random.randn(32, 7, 7, 64)
        out = flatten.forward(x)
        assert out.shape == (32, 7 * 7 * 64)

    def test_backward_shape(self) -> None:
        flatten = Flatten()
        x = np.random.randn(32, 7, 7, 64)
        out = flatten.forward(x)
        grad_out = np.random.randn(*out.shape)
        grad_x = flatten.backward(grad_out)
        assert grad_x.shape == x.shape

    def test_values_preserved(self) -> None:
        flatten = Flatten()
        x = np.arange(24).reshape(2, 2, 2, 3)
        out = flatten.forward(x)
        np.testing.assert_array_equal(out[0], x[0].flatten())


class TestBuildLenet:
    def test_returns_list(self) -> None:
        layers = build_lenet()
        assert isinstance(layers, list)
        assert len(layers) > 0

    def test_layer_types(self) -> None:
        layers = build_lenet()
        assert isinstance(layers[0], Conv2D)
        has_pool = any(isinstance(l, MaxPool2D) for l in layers)
        has_flatten = any(isinstance(l, Flatten) for l in layers)
        assert has_pool
        assert has_flatten

    def test_custom_num_classes(self) -> None:
        layers = build_lenet(num_classes=5)
        assert layers is not None


class TestForwardCNN:
    def test_simple_network(self) -> None:
        layers = [
            Conv2D(1, 4, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2),
            Flatten(),
        ]
        x = np.random.randn(2, 8, 8, 1)
        out = forward_cnn(layers, x)
        assert out.shape == (2, 4 * 4 * 4)

    def test_output_changes_with_input(self) -> None:
        layers = [Conv2D(1, 1, kernel_size=3, padding=1), ReLU()]
        x1 = np.random.randn(1, 4, 4, 1)
        x2 = np.random.randn(1, 4, 4, 1)
        out1 = forward_cnn(layers, x1)
        out2 = forward_cnn(layers, x2)
        assert not np.allclose(out1, out2)


class TestBackwardCNN:
    def test_backward_runs(self) -> None:
        layers = [
            Conv2D(1, 4, kernel_size=3, padding=1),
            ReLU(),
            Flatten(),
        ]
        x = np.random.randn(2, 4, 4, 1)
        out = forward_cnn(layers, x)
        grad_out = np.random.randn(*out.shape)
        backward_cnn(layers, grad_out)
        grad_W, grad_b = layers[0].get_gradients()
        assert grad_W is not None
        assert grad_b is not None


class TestCNNIntegration:
    def test_conv_pool_flatten(self) -> None:
        np.random.seed(42)
        x = np.random.randn(4, 8, 8, 1)
        layers = [
            Conv2D(1, 2, kernel_size=3, padding=0),
            ReLU(),
            MaxPool2D(pool_size=2),
            Flatten(),
        ]
        out = forward_cnn(layers, x)
        assert out.shape == (4, 3 * 3 * 2)

    def test_gradient_flow(self) -> None:
        np.random.seed(42)
        x = np.random.randn(2, 6, 6, 1)
        layers = [
            Conv2D(1, 2, kernel_size=3, padding=1),
            ReLU(),
            Flatten(),
        ]
        out = forward_cnn(layers, x)
        target = np.random.randn(*out.shape)
        loss_grad = 2 * (out - target)
        backward_cnn(layers, loss_grad)
        assert layers[0].get_gradients()[0] is not None
