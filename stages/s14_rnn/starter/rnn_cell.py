from __future__ import annotations

import numpy as np


def rnn_cell_forward(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Forward pass for a single RNN cell.

    Args:
        x_t: Input at time t, shape (batch, input_size)
        h_prev: Previous hidden state, shape (batch, hidden_size)
        W_xh: Input-to-hidden weights, shape (input_size, hidden_size)
        W_hh: Hidden-to-hidden weights, shape (hidden_size, hidden_size)
        b_h: Hidden bias, shape (hidden_size,)

    Returns:
        h_t: New hidden state, shape (batch, hidden_size)
        cache: Values needed for backward pass
    """
    # TODO:
    # z = x_t @ W_xh + h_prev @ W_hh + b_h
    # h_t = np.tanh(z)
    # cache = {'x_t': x_t, 'h_prev': h_prev, 'h_t': h_t, 'W_xh': W_xh, 'W_hh': W_hh}
    # return h_t, cache
    raise NotImplementedError


def rnn_cell_backward(
    grad_h: np.ndarray,
    cache: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass for a single RNN cell.

    Args:
        grad_h: Gradient w.r.t. h_t, shape (batch, hidden_size)
        cache: Cache from forward pass

    Returns:
        grad_x: Gradient w.r.t. x_t
        grad_h_prev: Gradient w.r.t. h_{t-1}
        grad_W_xh: Gradient w.r.t. W_xh
        grad_W_hh: Gradient w.r.t. W_hh
        grad_b_h: Gradient w.r.t. b_h
    """
    # TODO:
    # x_t = cache['x_t']
    # h_prev = cache['h_prev']
    # h_t = cache['h_t']
    # W_xh = cache['W_xh']
    # W_hh = cache['W_hh']
    #
    # dtanh = grad_h * (1 - h_t ** 2)
    #
    # grad_x = dtanh @ W_xh.T
    # grad_h_prev = dtanh @ W_hh.T
    # grad_W_xh = x_t.T @ dtanh
    # grad_W_hh = h_prev.T @ dtanh
    # grad_b_h = np.sum(dtanh, axis=0)
    #
    # return grad_x, grad_h_prev, grad_W_xh, grad_W_hh, grad_b_h
    raise NotImplementedError


def rnn_forward(
    x: np.ndarray,
    h_0: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Forward pass for full RNN over a sequence.

    Args:
        x: Input sequence, shape (batch, seq_length, input_size)
        h_0: Initial hidden state, shape (batch, hidden_size)
        W_xh, W_hh, b_h: RNN parameters

    Returns:
        h_all: All hidden states, shape (batch, seq_length, hidden_size)
        h_final: Final hidden state, shape (batch, hidden_size)
        caches: List of caches for backward pass
    """
    # TODO:
    # batch, seq_length, _ = x.shape
    # hidden_size = h_0.shape[1]
    #
    # h_all = np.zeros((batch, seq_length, hidden_size))
    # caches = []
    # h = h_0
    #
    # for t in range(seq_length):
    #     h, cache = rnn_cell_forward(x[:, t, :], h, W_xh, W_hh, b_h)
    #     h_all[:, t, :] = h
    #     caches.append(cache)
    #
    # return h_all, h, caches
    raise NotImplementedError


def rnn_backward(
    grad_h_all: np.ndarray,
    caches: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass for full RNN.

    Args:
        grad_h_all: Gradients for all hidden states, shape (batch, seq_length, hidden_size)
        caches: Caches from forward pass

    Returns:
        grad_x: Gradient w.r.t. input
        grad_W_xh: Accumulated gradient w.r.t. W_xh
        grad_W_hh: Accumulated gradient w.r.t. W_hh
        grad_b_h: Accumulated gradient w.r.t. b_h
    """
    # TODO:
    # batch, seq_length, hidden_size = grad_h_all.shape
    # input_size = caches[0]['x_t'].shape[1]
    #
    # grad_x = np.zeros((batch, seq_length, input_size))
    # grad_W_xh = np.zeros((input_size, hidden_size))
    # grad_W_hh = np.zeros((hidden_size, hidden_size))
    # grad_b_h = np.zeros(hidden_size)
    #
    # grad_h_next = np.zeros((batch, hidden_size))
    #
    # for t in reversed(range(seq_length)):
    #     grad_h = grad_h_all[:, t, :] + grad_h_next
    #     grad_x_t, grad_h_next, dW_xh, dW_hh, db_h = rnn_cell_backward(grad_h, caches[t])
    #     grad_x[:, t, :] = grad_x_t
    #     grad_W_xh += dW_xh
    #     grad_W_hh += dW_hh
    #     grad_b_h += db_h
    #
    # return grad_x, grad_W_xh, grad_W_hh, grad_b_h
    raise NotImplementedError


class VanillaRNN:
    """Vanilla RNN layer."""

    def __init__(self, input_size: int, hidden_size: int):
        """
        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
        """
        # TODO:
        # self.hidden_size = hidden_size
        # self.input_size = input_size
        #
        # scale = np.sqrt(2.0 / (input_size + hidden_size))
        # self.W_xh = np.random.randn(input_size, hidden_size) * scale
        # self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        # self.b_h = np.zeros(hidden_size)
        #
        # self.caches = None
        # self.grad_W_xh = None
        # self.grad_W_hh = None
        # self.grad_b_h = None
        raise NotImplementedError

    def forward(self, x: np.ndarray, h_0: np.ndarray | None = None) -> np.ndarray:
        """Process sequence."""
        # TODO:
        # batch = x.shape[0]
        # if h_0 is None:
        #     h_0 = np.zeros((batch, self.hidden_size))
        # h_all, _, self.caches = rnn_forward(x, h_0, self.W_xh, self.W_hh, self.b_h)
        # return h_all
        raise NotImplementedError

    def backward(self, grad_h_all: np.ndarray) -> np.ndarray:
        """Compute gradients."""
        # TODO:
        # grad_x, self.grad_W_xh, self.grad_W_hh, self.grad_b_h = rnn_backward(
        #     grad_h_all, self.caches
        # )
        # return grad_x
        raise NotImplementedError

    def get_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return parameters."""
        return self.W_xh, self.W_hh, self.b_h

    def set_params(self, W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> None:
        """Set parameters."""
        self.W_xh = W_xh
        self.W_hh = W_hh
        self.b_h = b_h

    def get_gradients(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return gradients."""
        return self.grad_W_xh, self.grad_W_hh, self.grad_b_h
