from __future__ import annotations

import numpy as np

from stages.s14_rnn.starter.rnn_cell import (
    VanillaRNN,
    rnn_backward,
    rnn_cell_backward,
    rnn_cell_forward,
    rnn_forward,
)


class TestRNNCellForward:
    def test_output_shape(self) -> None:
        np.random.seed(42)
        x = np.random.randn(4, 10)
        h_prev = np.random.randn(4, 8)
        W_xh = np.random.randn(10, 8)
        W_hh = np.random.randn(8, 8)
        b_h = np.zeros(8)
        h_t, _ = rnn_cell_forward(x, h_prev, W_xh, W_hh, b_h)
        assert h_t.shape == (4, 8)

    def test_tanh_bounds(self) -> None:
        np.random.seed(42)
        x = np.random.randn(4, 10) * 10
        h_prev = np.random.randn(4, 8) * 10
        W_xh = np.random.randn(10, 8)
        W_hh = np.random.randn(8, 8)
        b_h = np.zeros(8)
        h_t, _ = rnn_cell_forward(x, h_prev, W_xh, W_hh, b_h)
        assert np.all(h_t >= -1) and np.all(h_t <= 1)

    def test_cache_contents(self) -> None:
        np.random.seed(42)
        x = np.random.randn(2, 5)
        h_prev = np.random.randn(2, 4)
        W_xh = np.random.randn(5, 4)
        W_hh = np.random.randn(4, 4)
        b_h = np.zeros(4)
        _, cache = rnn_cell_forward(x, h_prev, W_xh, W_hh, b_h)
        assert 'x_t' in cache
        assert 'h_prev' in cache
        assert 'h_t' in cache


class TestRNNCellBackward:
    def test_gradient_shapes(self) -> None:
        np.random.seed(42)
        x = np.random.randn(4, 10)
        h_prev = np.random.randn(4, 8)
        W_xh = np.random.randn(10, 8)
        W_hh = np.random.randn(8, 8)
        b_h = np.zeros(8)
        h_t, cache = rnn_cell_forward(x, h_prev, W_xh, W_hh, b_h)
        grad_h = np.random.randn(4, 8)
        grad_x, grad_h_prev, grad_W_xh, grad_W_hh, grad_b_h = rnn_cell_backward(grad_h, cache)
        assert grad_x.shape == x.shape
        assert grad_h_prev.shape == h_prev.shape
        assert grad_W_xh.shape == W_xh.shape
        assert grad_W_hh.shape == W_hh.shape
        assert grad_b_h.shape == b_h.shape

    def test_numerical_gradient(self) -> None:
        np.random.seed(42)
        x = np.random.randn(2, 3)
        h_prev = np.random.randn(2, 4)
        W_xh = np.random.randn(3, 4)
        W_hh = np.random.randn(4, 4)
        b_h = np.zeros(4)
        h_t, cache = rnn_cell_forward(x, h_prev, W_xh, W_hh, b_h)
        grad_h = np.ones((2, 4))
        _, _, grad_W_xh, _, _ = rnn_cell_backward(grad_h, cache)
        eps = 1e-5
        numerical = np.zeros_like(W_xh)
        for i in range(W_xh.shape[0]):
            for j in range(W_xh.shape[1]):
                W_plus = W_xh.copy()
                W_plus[i, j] += eps
                h_plus, _ = rnn_cell_forward(x, h_prev, W_plus, W_hh, b_h)
                W_minus = W_xh.copy()
                W_minus[i, j] -= eps
                h_minus, _ = rnn_cell_forward(x, h_prev, W_minus, W_hh, b_h)
                numerical[i, j] = np.sum(h_plus - h_minus) / (2 * eps)
        np.testing.assert_allclose(grad_W_xh, numerical, rtol=1e-4, atol=1e-4)


class TestRNNForward:
    def test_output_shape(self) -> None:
        np.random.seed(42)
        x = np.random.randn(4, 10, 5)
        h_0 = np.zeros((4, 8))
        W_xh = np.random.randn(5, 8)
        W_hh = np.random.randn(8, 8)
        b_h = np.zeros(8)
        h_all, h_final, _ = rnn_forward(x, h_0, W_xh, W_hh, b_h)
        assert h_all.shape == (4, 10, 8)
        assert h_final.shape == (4, 8)

    def test_final_state_matches(self) -> None:
        np.random.seed(42)
        x = np.random.randn(2, 5, 3)
        h_0 = np.zeros((2, 4))
        W_xh = np.random.randn(3, 4)
        W_hh = np.random.randn(4, 4)
        b_h = np.zeros(4)
        h_all, h_final, _ = rnn_forward(x, h_0, W_xh, W_hh, b_h)
        np.testing.assert_array_equal(h_all[:, -1, :], h_final)


class TestRNNBackward:
    def test_gradient_shapes(self) -> None:
        np.random.seed(42)
        x = np.random.randn(4, 10, 5)
        h_0 = np.zeros((4, 8))
        W_xh = np.random.randn(5, 8)
        W_hh = np.random.randn(8, 8)
        b_h = np.zeros(8)
        h_all, _, caches = rnn_forward(x, h_0, W_xh, W_hh, b_h)
        grad_h_all = np.random.randn(*h_all.shape)
        grad_x, grad_W_xh, grad_W_hh, grad_b_h = rnn_backward(grad_h_all, caches)
        assert grad_x.shape == x.shape
        assert grad_W_xh.shape == W_xh.shape
        assert grad_W_hh.shape == W_hh.shape
        assert grad_b_h.shape == b_h.shape


class TestVanillaRNN:
    def test_initialization(self) -> None:
        rnn = VanillaRNN(10, 8)
        W_xh, W_hh, b_h = rnn.get_params()
        assert W_xh.shape == (10, 8)
        assert W_hh.shape == (8, 8)
        assert b_h.shape == (8,)

    def test_forward(self) -> None:
        rnn = VanillaRNN(5, 8)
        x = np.random.randn(4, 10, 5)
        h_all = rnn.forward(x)
        assert h_all.shape == (4, 10, 8)

    def test_backward(self) -> None:
        rnn = VanillaRNN(5, 8)
        x = np.random.randn(4, 10, 5)
        h_all = rnn.forward(x)
        grad_h_all = np.random.randn(*h_all.shape)
        grad_x = rnn.backward(grad_h_all)
        assert grad_x.shape == x.shape
        grads = rnn.get_gradients()
        assert grads[0].shape == (5, 8)

    def test_custom_initial_state(self) -> None:
        rnn = VanillaRNN(5, 8)
        x = np.random.randn(4, 10, 5)
        h_0 = np.ones((4, 8))
        h_all = rnn.forward(x, h_0)
        assert h_all.shape == (4, 10, 8)
