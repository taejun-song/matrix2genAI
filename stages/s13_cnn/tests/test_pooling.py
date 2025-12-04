from __future__ import annotations

import numpy as np

from stages.s13_cnn.starter.pooling import (
    AvgPool2D,
    MaxPool2D,
    avg_pool2d_backward,
    avg_pool2d_forward,
    max_pool2d_backward,
    max_pool2d_forward,
)


class TestMaxPool2dForward:
    def test_output_shape(self) -> None:
        x = np.random.randn(2, 8, 8, 3)
        out, _ = max_pool2d_forward(x, pool_size=2)
        assert out.shape == (2, 4, 4, 3)

    def test_max_values(self) -> None:
        x = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]).reshape(1, 4, 4, 1).astype(float)
        out, _ = max_pool2d_forward(x, pool_size=2)
        expected = np.array([[6, 8], [14, 16]]).reshape(1, 2, 2, 1)
        np.testing.assert_array_equal(out, expected)

    def test_stride1(self) -> None:
        x = np.random.randn(1, 5, 5, 1)
        out, _ = max_pool2d_forward(x, pool_size=2, stride=1)
        assert out.shape == (1, 4, 4, 1)

    def test_multi_channel(self) -> None:
        x = np.random.randn(1, 4, 4, 3)
        out, indices = max_pool2d_forward(x, pool_size=2)
        assert out.shape == (1, 2, 2, 3)
        assert indices.shape == (1, 2, 2, 3, 2)


class TestMaxPool2dBackward:
    def test_gradient_shape(self) -> None:
        x = np.random.randn(2, 8, 8, 3)
        out, indices = max_pool2d_forward(x, pool_size=2)
        grad_out = np.random.randn(*out.shape)
        grad_x = max_pool2d_backward(grad_out, indices, x.shape, pool_size=2)
        assert grad_x.shape == x.shape

    def test_gradient_routing(self) -> None:
        x = np.array([
            [1, 2],
            [3, 4]
        ]).reshape(1, 2, 2, 1).astype(float)
        out, indices = max_pool2d_forward(x, pool_size=2)
        grad_out = np.array([[1.0]]).reshape(1, 1, 1, 1)
        grad_x = max_pool2d_backward(grad_out, indices, x.shape, pool_size=2)
        expected = np.array([[0, 0], [0, 1]]).reshape(1, 2, 2, 1)
        np.testing.assert_array_equal(grad_x, expected)


class TestMaxPool2DClass:
    def test_forward(self) -> None:
        pool = MaxPool2D(pool_size=2)
        x = np.random.randn(2, 8, 8, 4)
        out = pool.forward(x)
        assert out.shape == (2, 4, 4, 4)

    def test_backward(self) -> None:
        pool = MaxPool2D(pool_size=2)
        x = np.random.randn(2, 8, 8, 4)
        out = pool.forward(x)
        grad_out = np.random.randn(*out.shape)
        grad_x = pool.backward(grad_out)
        assert grad_x.shape == x.shape


class TestAvgPool2dForward:
    def test_output_shape(self) -> None:
        x = np.random.randn(2, 8, 8, 3)
        out = avg_pool2d_forward(x, pool_size=2)
        assert out.shape == (2, 4, 4, 3)

    def test_average_values(self) -> None:
        x = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]).reshape(1, 4, 4, 1).astype(float)
        out = avg_pool2d_forward(x, pool_size=2)
        expected = np.array([[3.5, 5.5], [11.5, 13.5]]).reshape(1, 2, 2, 1)
        np.testing.assert_array_equal(out, expected)

    def test_stride1(self) -> None:
        x = np.random.randn(1, 5, 5, 1)
        out = avg_pool2d_forward(x, pool_size=2, stride=1)
        assert out.shape == (1, 4, 4, 1)


class TestAvgPool2dBackward:
    def test_gradient_shape(self) -> None:
        x = np.random.randn(2, 8, 8, 3)
        out = avg_pool2d_forward(x, pool_size=2)
        grad_out = np.random.randn(*out.shape)
        grad_x = avg_pool2d_backward(grad_out, x.shape, pool_size=2)
        assert grad_x.shape == x.shape

    def test_gradient_distribution(self) -> None:
        x = np.zeros((1, 2, 2, 1))
        out = avg_pool2d_forward(x, pool_size=2)
        grad_out = np.array([[4.0]]).reshape(1, 1, 1, 1)
        grad_x = avg_pool2d_backward(grad_out, x.shape, pool_size=2)
        expected = np.ones((1, 2, 2, 1))
        np.testing.assert_array_equal(grad_x, expected)


class TestAvgPool2DClass:
    def test_forward(self) -> None:
        pool = AvgPool2D(pool_size=2)
        x = np.random.randn(2, 8, 8, 4)
        out = pool.forward(x)
        assert out.shape == (2, 4, 4, 4)

    def test_backward(self) -> None:
        pool = AvgPool2D(pool_size=2)
        x = np.random.randn(2, 8, 8, 4)
        out = pool.forward(x)
        grad_out = np.random.randn(*out.shape)
        grad_x = pool.backward(grad_out)
        assert grad_x.shape == x.shape
