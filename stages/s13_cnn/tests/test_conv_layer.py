from __future__ import annotations

import numpy as np

from stages.s13_cnn.starter.conv_layer import Conv2D, conv2d_backward, conv2d_forward


class TestConv2dForward:
    def test_output_shape(self) -> None:
        x = np.random.randn(2, 8, 8, 3)
        W = np.random.randn(3, 3, 3, 16)
        b = np.zeros(16)
        out, _ = conv2d_forward(x, W, b, stride=1, padding=0)
        assert out.shape == (2, 6, 6, 16)

    def test_with_padding(self) -> None:
        x = np.random.randn(1, 5, 5, 1)
        W = np.random.randn(3, 3, 1, 4)
        b = np.zeros(4)
        out, _ = conv2d_forward(x, W, b, stride=1, padding=1)
        assert out.shape == (1, 5, 5, 4)

    def test_stride2(self) -> None:
        x = np.random.randn(1, 8, 8, 1)
        W = np.random.randn(2, 2, 1, 8)
        b = np.zeros(8)
        out, _ = conv2d_forward(x, W, b, stride=2, padding=0)
        assert out.shape == (1, 4, 4, 8)

    def test_bias_addition(self) -> None:
        x = np.zeros((1, 3, 3, 1))
        W = np.zeros((3, 3, 1, 2))
        b = np.array([1.0, 2.0])
        out, _ = conv2d_forward(x, W, b, stride=1, padding=0)
        assert out.shape == (1, 1, 1, 2)
        np.testing.assert_array_equal(out[0, 0, 0], [1.0, 2.0])


class TestConv2dBackward:
    def test_gradient_shapes(self) -> None:
        np.random.seed(42)
        x = np.random.randn(2, 8, 8, 3)
        W = np.random.randn(3, 3, 3, 16)
        b = np.zeros(16)
        out, col = conv2d_forward(x, W, b, stride=1, padding=0)
        grad_out = np.random.randn(*out.shape)
        grad_x, grad_W, grad_b = conv2d_backward(grad_out, col, x.shape, W, stride=1, padding=0)
        assert grad_x.shape == x.shape
        assert grad_W.shape == W.shape
        assert grad_b.shape == b.shape

    def test_numerical_gradient(self) -> None:
        np.random.seed(42)
        x = np.random.randn(1, 4, 4, 1)
        W = np.random.randn(2, 2, 1, 1)
        b = np.zeros(1)
        eps = 1e-5
        out, col = conv2d_forward(x, W, b)
        grad_out = np.ones_like(out)
        _, grad_W, _ = conv2d_backward(grad_out, col, x.shape, W)
        numerical_grad = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_plus = W.copy()
                W_plus[i, j, 0, 0] += eps
                out_plus, _ = conv2d_forward(x, W_plus, b)
                W_minus = W.copy()
                W_minus[i, j, 0, 0] -= eps
                out_minus, _ = conv2d_forward(x, W_minus, b)
                numerical_grad[i, j, 0, 0] = np.sum(out_plus - out_minus) / (2 * eps)
        np.testing.assert_allclose(grad_W, numerical_grad, rtol=1e-4, atol=1e-4)


class TestConv2DClass:
    def test_initialization(self) -> None:
        layer = Conv2D(3, 16, kernel_size=3, stride=1, padding=1)
        W, b = layer.get_params()
        assert W.shape == (3, 3, 3, 16)
        assert b.shape == (16,)

    def test_forward(self) -> None:
        layer = Conv2D(1, 8, kernel_size=3, stride=1, padding=0)
        x = np.random.randn(2, 10, 10, 1)
        out = layer.forward(x)
        assert out.shape == (2, 8, 8, 8)

    def test_backward(self) -> None:
        layer = Conv2D(3, 4, kernel_size=3, stride=1, padding=1)
        x = np.random.randn(2, 5, 5, 3)
        out = layer.forward(x)
        grad_out = np.random.randn(*out.shape)
        grad_x = layer.backward(grad_out)
        assert grad_x.shape == x.shape
        grad_W, grad_b = layer.get_gradients()
        assert grad_W.shape == (3, 3, 3, 4)
        assert grad_b.shape == (4,)

    def test_set_params(self) -> None:
        layer = Conv2D(1, 1, kernel_size=2)
        new_W = np.ones((2, 2, 1, 1))
        new_b = np.array([5.0])
        layer.set_params(new_W, new_b)
        W, b = layer.get_params()
        np.testing.assert_array_equal(W, new_W)
        np.testing.assert_array_equal(b, new_b)


class TestConvPatterns:
    def test_edge_detection(self) -> None:
        layer = Conv2D(1, 1, kernel_size=3, padding=0)
        edge_filter = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ]).reshape(3, 3, 1, 1).astype(float)
        layer.set_params(edge_filter, np.array([0.0]))
        x = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ]).reshape(1, 5, 5, 1).astype(float)
        out = layer.forward(x)
        assert out.shape == (1, 3, 3, 1)
        assert np.all(out[:, :, 0, :] < 0)
