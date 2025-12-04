from __future__ import annotations

import numpy as np

from stages.s14_rnn.starter.lstm_cell import (
    LSTM,
    lstm_backward,
    lstm_cell_backward,
    lstm_cell_forward,
    lstm_forward,
)


class TestLSTMCellForward:
    def test_output_shapes(self) -> None:
        np.random.seed(42)
        batch, input_size, hidden_size = 4, 10, 8
        concat_size = hidden_size + input_size
        x = np.random.randn(batch, input_size)
        h_prev = np.random.randn(batch, hidden_size)
        c_prev = np.random.randn(batch, hidden_size)
        W_f = np.random.randn(concat_size, hidden_size)
        W_i = np.random.randn(concat_size, hidden_size)
        W_c = np.random.randn(concat_size, hidden_size)
        W_o = np.random.randn(concat_size, hidden_size)
        b_f = np.zeros(hidden_size)
        b_i = np.zeros(hidden_size)
        b_c = np.zeros(hidden_size)
        b_o = np.zeros(hidden_size)
        h_t, c_t, _ = lstm_cell_forward(x, h_prev, c_prev, W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o)
        assert h_t.shape == (batch, hidden_size)
        assert c_t.shape == (batch, hidden_size)

    def test_gate_values_bounded(self) -> None:
        np.random.seed(42)
        batch, input_size, hidden_size = 2, 5, 4
        concat_size = hidden_size + input_size
        x = np.random.randn(batch, input_size) * 10
        h_prev = np.random.randn(batch, hidden_size) * 10
        c_prev = np.random.randn(batch, hidden_size)
        W_f = np.random.randn(concat_size, hidden_size)
        W_i = np.random.randn(concat_size, hidden_size)
        W_c = np.random.randn(concat_size, hidden_size)
        W_o = np.random.randn(concat_size, hidden_size)
        b_f = np.zeros(hidden_size)
        b_i = np.zeros(hidden_size)
        b_c = np.zeros(hidden_size)
        b_o = np.zeros(hidden_size)
        _, _, cache = lstm_cell_forward(x, h_prev, c_prev, W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o)
        assert np.all(cache['f_t'] >= 0) and np.all(cache['f_t'] <= 1)
        assert np.all(cache['i_t'] >= 0) and np.all(cache['i_t'] <= 1)
        assert np.all(cache['o_t'] >= 0) and np.all(cache['o_t'] <= 1)


class TestLSTMCellBackward:
    def test_gradient_shapes(self) -> None:
        np.random.seed(42)
        batch, input_size, hidden_size = 4, 10, 8
        concat_size = hidden_size + input_size
        x = np.random.randn(batch, input_size)
        h_prev = np.random.randn(batch, hidden_size)
        c_prev = np.random.randn(batch, hidden_size)
        W_f = np.random.randn(concat_size, hidden_size)
        W_i = np.random.randn(concat_size, hidden_size)
        W_c = np.random.randn(concat_size, hidden_size)
        W_o = np.random.randn(concat_size, hidden_size)
        b_f = np.zeros(hidden_size)
        b_i = np.zeros(hidden_size)
        b_c = np.zeros(hidden_size)
        b_o = np.zeros(hidden_size)
        h_t, c_t, cache = lstm_cell_forward(x, h_prev, c_prev, W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o)
        grad_h = np.random.randn(batch, hidden_size)
        grad_c = np.random.randn(batch, hidden_size)
        grad_x, grad_h_prev, grad_c_prev, grad_params = lstm_cell_backward(grad_h, grad_c, cache)
        assert grad_x.shape == x.shape
        assert grad_h_prev.shape == h_prev.shape
        assert grad_c_prev.shape == c_prev.shape
        assert grad_params['W_f'].shape == W_f.shape
        assert grad_params['b_f'].shape == b_f.shape


class TestLSTMForward:
    def test_output_shapes(self) -> None:
        np.random.seed(42)
        batch, seq_len, input_size, hidden_size = 4, 10, 5, 8
        concat_size = hidden_size + input_size
        x = np.random.randn(batch, seq_len, input_size)
        h_0 = np.zeros((batch, hidden_size))
        c_0 = np.zeros((batch, hidden_size))
        params = {
            'W_f': np.random.randn(concat_size, hidden_size),
            'b_f': np.ones(hidden_size),
            'W_i': np.random.randn(concat_size, hidden_size),
            'b_i': np.zeros(hidden_size),
            'W_c': np.random.randn(concat_size, hidden_size),
            'b_c': np.zeros(hidden_size),
            'W_o': np.random.randn(concat_size, hidden_size),
            'b_o': np.zeros(hidden_size),
        }
        h_all, h_final, c_final, _ = lstm_forward(x, h_0, c_0, params)
        assert h_all.shape == (batch, seq_len, hidden_size)
        assert h_final.shape == (batch, hidden_size)
        assert c_final.shape == (batch, hidden_size)


class TestLSTMBackward:
    def test_gradient_shapes(self) -> None:
        np.random.seed(42)
        batch, seq_len, input_size, hidden_size = 4, 5, 3, 4
        concat_size = hidden_size + input_size
        x = np.random.randn(batch, seq_len, input_size)
        h_0 = np.zeros((batch, hidden_size))
        c_0 = np.zeros((batch, hidden_size))
        params = {
            'W_f': np.random.randn(concat_size, hidden_size),
            'b_f': np.ones(hidden_size),
            'W_i': np.random.randn(concat_size, hidden_size),
            'b_i': np.zeros(hidden_size),
            'W_c': np.random.randn(concat_size, hidden_size),
            'b_c': np.zeros(hidden_size),
            'W_o': np.random.randn(concat_size, hidden_size),
            'b_o': np.zeros(hidden_size),
        }
        h_all, _, _, caches = lstm_forward(x, h_0, c_0, params)
        grad_h_all = np.random.randn(*h_all.shape)
        grad_x, grad_params = lstm_backward(grad_h_all, caches)
        assert grad_x.shape == x.shape
        assert grad_params['W_f'].shape == params['W_f'].shape


class TestLSTMClass:
    def test_initialization(self) -> None:
        lstm = LSTM(10, 8)
        params = lstm.get_params()
        assert params['W_f'].shape == (18, 8)
        assert params['b_f'].shape == (8,)
        np.testing.assert_array_equal(params['b_f'], np.ones(8))

    def test_forward(self) -> None:
        lstm = LSTM(5, 8)
        x = np.random.randn(4, 10, 5)
        h_all = lstm.forward(x)
        assert h_all.shape == (4, 10, 8)

    def test_backward(self) -> None:
        lstm = LSTM(5, 8)
        x = np.random.randn(4, 10, 5)
        h_all = lstm.forward(x)
        grad_h_all = np.random.randn(*h_all.shape)
        grad_x = lstm.backward(grad_h_all)
        assert grad_x.shape == x.shape
        grads = lstm.get_gradients()
        assert 'W_f' in grads

    def test_set_params(self) -> None:
        lstm = LSTM(5, 8)
        params = lstm.get_params()
        params['b_f'] = np.zeros(8)
        lstm.set_params(params)
        new_params = lstm.get_params()
        np.testing.assert_array_equal(new_params['b_f'], np.zeros(8))
