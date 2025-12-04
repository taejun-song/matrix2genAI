# Stage 14: Recurrent Neural Networks - Specifications

## Module: rnn_cell.py

### rnn_cell_forward
```python
def rnn_cell_forward(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray
) -> tuple[np.ndarray, dict]:
    """
    Forward pass for a single RNN cell.

    h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b_h)

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
```

### rnn_cell_backward
```python
def rnn_cell_backward(
    grad_h: np.ndarray,
    cache: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass for a single RNN cell.

    Args:
        grad_h: Gradient w.r.t. h_t, shape (batch, hidden_size)
        cache: Cache from forward pass

    Returns:
        grad_x: Gradient w.r.t. x_t, shape (batch, input_size)
        grad_h_prev: Gradient w.r.t. h_{t-1}, shape (batch, hidden_size)
        grad_W_xh: Gradient w.r.t. W_xh
        grad_W_hh: Gradient w.r.t. W_hh
        grad_b_h: Gradient w.r.t. b_h
    """
```

### rnn_forward
```python
def rnn_forward(
    x: np.ndarray,
    h_0: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray
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
```

### rnn_backward
```python
def rnn_backward(
    grad_h_all: np.ndarray,
    caches: list
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass for full RNN.

    Args:
        grad_h_all: Gradients for all hidden states, shape (batch, seq_length, hidden_size)
        caches: Caches from forward pass

    Returns:
        grad_x: Gradient w.r.t. input, shape (batch, seq_length, input_size)
        grad_W_xh: Accumulated gradient w.r.t. W_xh
        grad_W_hh: Accumulated gradient w.r.t. W_hh
        grad_b_h: Accumulated gradient w.r.t. b_h
    """
```

### VanillaRNN (class)
```python
class VanillaRNN:
    """
    Vanilla RNN layer.

    Attributes:
        W_xh: Input-to-hidden weights
        W_hh: Hidden-to-hidden weights
        b_h: Hidden bias
        hidden_size: Size of hidden state

    Methods:
        forward(x, h_0=None): Process sequence
        backward(grad_h_all): Compute gradients
        get_params(): Return parameters
        set_params(W_xh, W_hh, b_h): Set parameters
        get_gradients(): Return gradients
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
        """
```

---

## Module: lstm_cell.py

### lstm_cell_forward
```python
def lstm_cell_forward(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    c_prev: np.ndarray,
    W_f: np.ndarray, b_f: np.ndarray,
    W_i: np.ndarray, b_i: np.ndarray,
    W_c: np.ndarray, b_c: np.ndarray,
    W_o: np.ndarray, b_o: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Forward pass for a single LSTM cell.

    f_t = sigmoid(W_f @ [h_prev, x_t] + b_f)  # Forget gate
    i_t = sigmoid(W_i @ [h_prev, x_t] + b_i)  # Input gate
    c_tilde = tanh(W_c @ [h_prev, x_t] + b_c)  # Candidate
    o_t = sigmoid(W_o @ [h_prev, x_t] + b_o)  # Output gate
    c_t = f_t * c_prev + i_t * c_tilde
    h_t = o_t * tanh(c_t)

    Args:
        x_t: Input, shape (batch, input_size)
        h_prev: Previous hidden state, shape (batch, hidden_size)
        c_prev: Previous cell state, shape (batch, hidden_size)
        W_f, b_f: Forget gate parameters
        W_i, b_i: Input gate parameters
        W_c, b_c: Candidate parameters
        W_o, b_o: Output gate parameters

    Returns:
        h_t: New hidden state
        c_t: New cell state
        cache: Values for backward pass
    """
```

### lstm_cell_backward
```python
def lstm_cell_backward(
    grad_h: np.ndarray,
    grad_c: np.ndarray,
    cache: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Backward pass for a single LSTM cell.

    Args:
        grad_h: Gradient w.r.t. h_t
        grad_c: Gradient w.r.t. c_t (from next time step)
        cache: Cache from forward pass

    Returns:
        grad_x: Gradient w.r.t. x_t
        grad_h_prev: Gradient w.r.t. h_{t-1}
        grad_c_prev: Gradient w.r.t. c_{t-1}
        grad_params: Dict with gradients for all weights and biases
    """
```

### lstm_forward
```python
def lstm_forward(
    x: np.ndarray,
    h_0: np.ndarray,
    c_0: np.ndarray,
    params: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Forward pass for full LSTM over a sequence.

    Args:
        x: Input sequence, shape (batch, seq_length, input_size)
        h_0: Initial hidden state
        c_0: Initial cell state
        params: Dict containing all LSTM parameters

    Returns:
        h_all: All hidden states, shape (batch, seq_length, hidden_size)
        h_final: Final hidden state
        c_final: Final cell state
        caches: List of caches
    """
```

### lstm_backward
```python
def lstm_backward(
    grad_h_all: np.ndarray,
    caches: list
) -> tuple[np.ndarray, dict]:
    """
    Backward pass for full LSTM.

    Args:
        grad_h_all: Gradients for all hidden states
        caches: Caches from forward pass

    Returns:
        grad_x: Gradient w.r.t. input
        grad_params: Dict with accumulated gradients for all parameters
    """
```

### LSTM (class)
```python
class LSTM:
    """
    LSTM layer.

    Attributes:
        hidden_size: Hidden dimension
        W_f, W_i, W_c, W_o: Gate weight matrices
        b_f, b_i, b_c, b_o: Gate biases

    Methods:
        forward(x, h_0=None, c_0=None): Process sequence
        backward(grad_h_all): Compute gradients
        get_params(): Return all parameters as dict
        set_params(params): Set parameters from dict
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize LSTM.

        Note: Forget gate bias initialized to 1.0 for better gradient flow.
        """
```

---

## Module: gru_cell.py

### gru_cell_forward
```python
def gru_cell_forward(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    W_z: np.ndarray, b_z: np.ndarray,
    W_r: np.ndarray, b_r: np.ndarray,
    W_h: np.ndarray, b_h: np.ndarray
) -> tuple[np.ndarray, dict]:
    """
    Forward pass for a single GRU cell.

    z_t = sigmoid(W_z @ [h_prev, x_t] + b_z)  # Update gate
    r_t = sigmoid(W_r @ [h_prev, x_t] + b_r)  # Reset gate
    h_tilde = tanh(W_h @ [r_t * h_prev, x_t] + b_h)  # Candidate
    h_t = (1 - z_t) * h_prev + z_t * h_tilde

    Args:
        x_t: Input, shape (batch, input_size)
        h_prev: Previous hidden state, shape (batch, hidden_size)
        W_z, b_z: Update gate parameters
        W_r, b_r: Reset gate parameters
        W_h, b_h: Candidate parameters

    Returns:
        h_t: New hidden state
        cache: Values for backward pass
    """
```

### gru_cell_backward
```python
def gru_cell_backward(
    grad_h: np.ndarray,
    cache: dict
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Backward pass for a single GRU cell.

    Args:
        grad_h: Gradient w.r.t. h_t
        cache: Cache from forward pass

    Returns:
        grad_x: Gradient w.r.t. x_t
        grad_h_prev: Gradient w.r.t. h_{t-1}
        grad_params: Dict with gradients for weights and biases
    """
```

### gru_forward
```python
def gru_forward(
    x: np.ndarray,
    h_0: np.ndarray,
    params: dict
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Forward pass for full GRU over a sequence.

    Args:
        x: Input sequence, shape (batch, seq_length, input_size)
        h_0: Initial hidden state
        params: Dict containing all GRU parameters

    Returns:
        h_all: All hidden states
        h_final: Final hidden state
        caches: List of caches
    """
```

### gru_backward
```python
def gru_backward(
    grad_h_all: np.ndarray,
    caches: list
) -> tuple[np.ndarray, dict]:
    """
    Backward pass for full GRU.

    Args:
        grad_h_all: Gradients for all hidden states
        caches: Caches from forward pass

    Returns:
        grad_x: Gradient w.r.t. input
        grad_params: Dict with accumulated gradients
    """
```

### GRU (class)
```python
class GRU:
    """
    GRU layer.

    Attributes:
        hidden_size: Hidden dimension
        W_z, W_r, W_h: Gate weight matrices
        b_z, b_r, b_h: Gate biases

    Methods:
        forward(x, h_0=None): Process sequence
        backward(grad_h_all): Compute gradients
        get_params(): Return all parameters
        set_params(params): Set parameters
    """

    def __init__(self, input_size: int, hidden_size: int):
        """Initialize GRU layer."""
```

---

## Module: sequence_utils.py

### clip_gradients
```python
def clip_gradients(
    gradients: list[np.ndarray],
    max_norm: float
) -> list[np.ndarray]:
    """
    Clip gradients by global norm.

    If ||gradients|| > max_norm, scale all gradients by max_norm / ||gradients||.

    Args:
        gradients: List of gradient arrays
        max_norm: Maximum allowed global norm

    Returns:
        clipped: List of clipped gradients

    Example:
        >>> grads = [np.array([3.0, 4.0])]  # norm = 5
        >>> clipped = clip_gradients(grads, max_norm=2.5)
        >>> np.linalg.norm(clipped[0])  # 2.5
    """
```

### sequence_loss
```python
def sequence_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray | None = None
) -> float:
    """
    Compute cross-entropy loss over sequences.

    Args:
        predictions: Predicted logits, shape (batch, seq_length, vocab_size)
        targets: Target indices, shape (batch, seq_length)
        mask: Optional mask for padding, shape (batch, seq_length)

    Returns:
        loss: Average cross-entropy loss
    """
```

### generate_sequence
```python
def generate_sequence(
    model,
    seed_sequence: np.ndarray,
    length: int,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Generate sequence by sampling from model predictions.

    Args:
        model: RNN/LSTM/GRU model with forward method
        seed_sequence: Initial input, shape (1, seed_length, input_size)
        length: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)

    Returns:
        generated: Generated sequence, shape (1, length, input_size)
    """
```

### create_sequences
```python
def create_sequences(
    data: np.ndarray,
    seq_length: int,
    stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create input/target pairs for sequence prediction.

    Args:
        data: Source data, shape (total_length, features)
        seq_length: Length of each sequence
        stride: Step between sequences

    Returns:
        X: Input sequences, shape (n_sequences, seq_length, features)
        y: Target values (next step), shape (n_sequences, features)
    """
```
