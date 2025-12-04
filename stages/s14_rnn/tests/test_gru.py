from __future__ import annotations

import numpy as np

from stages.s14_rnn.starter.gru_cell import (
    GRU,
    gru_backward,
    gru_cell_backward,
    gru_cell_forward,
    gru_forward,
)


class TestGRUCellForward:
    def test_output_shape(self) -> None:
        np.random.seed(42)
        batch, input_size, hidden_size = 4, 10, 8
        concat_size = hidden_size + input_size
        x = np.random.randn(batch, input_size)
        h_prev = np.random.randn(batch, hidden_size)
        W_z = np.random.randn(concat_size, hidden_size)
        W_r = np.random.randn(concat_size, hidden_size)
        W_h = np.random.randn(concat_size, hidden_size)
        b_z = np.zeros(hidden_size)
        b_r = np.zeros(hidden_size)
        b_h = np.zeros(hidden_size)
        h_t, _ = gru_cell_forward(x, h_prev, W_z, b_z, W_r, b_r, W_h, b_h)
        assert h_t.shape == (batch, hidden_size)

    def test_gate_values_bounded(self) -> None:
        np.random.seed(42)
        batch, input_size, hidden_size = 2, 5, 4
        concat_size = hidden_size + input_size
        x = np.random.randn(batch, input_size) * 10
        h_prev = np.random.randn(batch, hidden_size) * 10
        W_z = np.random.randn(concat_size, hidden_size)
        W_r = np.random.randn(concat_size, hidden_size)
        W_h = np.random.randn(concat_size, hidden_size)
        b_z = np.zeros(hidden_size)
        b_r = np.zeros(hidden_size)
        b_h = np.zeros(hidden_size)
        _, cache = gru_cell_forward(x, h_prev, W_z, b_z, W_r, b_r, W_h, b_h)
        assert np.all(cache['z_t'] >= 0) and np.all(cache['z_t'] <= 1)
        assert np.all(cache['r_t'] >= 0) and np.all(cache['r_t'] <= 1)

    def test_cache_contents(self) -> None:
        np.random.seed(42)
        batch, input_size, hidden_size = 2, 5, 4
        concat_size = hidden_size + input_size
        x = np.random.randn(batch, input_size)
        h_prev = np.random.randn(batch, hidden_size)
        W_z = np.random.randn(concat_size, hidden_size)
        W_r = np.random.randn(concat_size, hidden_size)
        W_h = np.random.randn(concat_size, hidden_size)
        b_z = np.zeros(hidden_size)
        b_r = np.zeros(hidden_size)
        b_h = np.zeros(hidden_size)
        _, cache = gru_cell_forward(x, h_prev, W_z, b_z, W_r, b_r, W_h, b_h)
        assert 'z_t' in cache
        assert 'r_t' in cache
        assert 'h_tilde' in cache


class TestGRUCellBackward:
    def test_gradient_shapes(self) -> None:
        np.random.seed(42)
        batch, input_size, hidden_size = 4, 10, 8
        concat_size = hidden_size + input_size
        x = np.random.randn(batch, input_size)
        h_prev = np.random.randn(batch, hidden_size)
        W_z = np.random.randn(concat_size, hidden_size)
        W_r = np.random.randn(concat_size, hidden_size)
        W_h = np.random.randn(concat_size, hidden_size)
        b_z = np.zeros(hidden_size)
        b_r = np.zeros(hidden_size)
        b_h = np.zeros(hidden_size)
        h_t, cache = gru_cell_forward(x, h_prev, W_z, b_z, W_r, b_r, W_h, b_h)
        grad_h = np.random.randn(batch, hidden_size)
        grad_x, grad_h_prev, grad_params = gru_cell_backward(grad_h, cache)
        assert grad_x.shape == x.shape
        assert grad_h_prev.shape == h_prev.shape
        assert grad_params['W_z'].shape == W_z.shape
        assert grad_params['b_z'].shape == b_z.shape


class TestGRUForward:
    def test_output_shapes(self) -> None:
        np.random.seed(42)
        batch, seq_len, input_size, hidden_size = 4, 10, 5, 8
        concat_size = hidden_size + input_size
        x = np.random.randn(batch, seq_len, input_size)
        h_0 = np.zeros((batch, hidden_size))
        params = {
            'W_z': np.random.randn(concat_size, hidden_size),
            'b_z': np.zeros(hidden_size),
            'W_r': np.random.randn(concat_size, hidden_size),
            'b_r': np.zeros(hidden_size),
            'W_h': np.random.randn(concat_size, hidden_size),
            'b_h': np.zeros(hidden_size),
        }
        h_all, h_final, _ = gru_forward(x, h_0, params)
        assert h_all.shape == (batch, seq_len, hidden_size)
        assert h_final.shape == (batch, hidden_size)

    def test_final_matches_last(self) -> None:
        np.random.seed(42)
        batch, seq_len, input_size, hidden_size = 2, 5, 3, 4
        concat_size = hidden_size + input_size
        x = np.random.randn(batch, seq_len, input_size)
        h_0 = np.zeros((batch, hidden_size))
        params = {
            'W_z': np.random.randn(concat_size, hidden_size),
            'b_z': np.zeros(hidden_size),
            'W_r': np.random.randn(concat_size, hidden_size),
            'b_r': np.zeros(hidden_size),
            'W_h': np.random.randn(concat_size, hidden_size),
            'b_h': np.zeros(hidden_size),
        }
        h_all, h_final, _ = gru_forward(x, h_0, params)
        np.testing.assert_array_equal(h_all[:, -1, :], h_final)


class TestGRUBackward:
    def test_gradient_shapes(self) -> None:
        np.random.seed(42)
        batch, seq_len, input_size, hidden_size = 4, 5, 3, 4
        concat_size = hidden_size + input_size
        x = np.random.randn(batch, seq_len, input_size)
        h_0 = np.zeros((batch, hidden_size))
        params = {
            'W_z': np.random.randn(concat_size, hidden_size),
            'b_z': np.zeros(hidden_size),
            'W_r': np.random.randn(concat_size, hidden_size),
            'b_r': np.zeros(hidden_size),
            'W_h': np.random.randn(concat_size, hidden_size),
            'b_h': np.zeros(hidden_size),
        }
        h_all, _, caches = gru_forward(x, h_0, params)
        grad_h_all = np.random.randn(*h_all.shape)
        grad_x, grad_params = gru_backward(grad_h_all, caches)
        assert grad_x.shape == x.shape
        assert grad_params['W_z'].shape == params['W_z'].shape


class TestGRUClass:
    def test_initialization(self) -> None:
        gru = GRU(10, 8)
        params = gru.get_params()
        assert params['W_z'].shape == (18, 8)
        assert params['b_z'].shape == (8,)

    def test_forward(self) -> None:
        gru = GRU(5, 8)
        x = np.random.randn(4, 10, 5)
        h_all = gru.forward(x)
        assert h_all.shape == (4, 10, 8)

    def test_backward(self) -> None:
        gru = GRU(5, 8)
        x = np.random.randn(4, 10, 5)
        h_all = gru.forward(x)
        grad_h_all = np.random.randn(*h_all.shape)
        grad_x = gru.backward(grad_h_all)
        assert grad_x.shape == x.shape
        grads = gru.get_gradients()
        assert 'W_z' in grads

    def test_fewer_params_than_lstm(self) -> None:
        from stages.s14_rnn.starter.lstm_cell import LSTM
        gru = GRU(10, 8)
        lstm = LSTM(10, 8)
        gru_params = sum(p.size for p in gru.get_params().values())
        lstm_params = sum(p.size for p in lstm.get_params().values())
        assert gru_params < lstm_params
