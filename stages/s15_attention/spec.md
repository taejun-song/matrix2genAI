# Stage 15: Attention Mechanisms - Specifications

## Module: attention_core.py

### compute_attention_scores
```python
def compute_attention_scores(
    Q: np.ndarray,
    K: np.ndarray,
    scale: bool = True
) -> np.ndarray:
    """
    Compute attention scores between queries and keys.

    scores = QK^T / √d_k  (if scale=True)

    Args:
        Q: Queries, shape (..., seq_q, d_k)
        K: Keys, shape (..., seq_k, d_k)
        scale: Whether to scale by √d_k

    Returns:
        scores: Shape (..., seq_q, seq_k)

    Example:
        >>> Q = np.random.randn(2, 4, 64)  # batch=2, seq=4, d_k=64
        >>> K = np.random.randn(2, 4, 64)
        >>> scores = compute_attention_scores(Q, K)
        >>> scores.shape
        (2, 4, 4)
    """
```

### apply_attention_mask
```python
def apply_attention_mask(
    scores: np.ndarray,
    mask: np.ndarray,
    mask_value: float = -1e9
) -> np.ndarray:
    """
    Apply mask to attention scores.

    Args:
        scores: Attention scores, shape (..., seq_q, seq_k)
        mask: Boolean mask, shape broadcastable to scores
              True = attend, False = mask out
        mask_value: Value to use for masked positions

    Returns:
        masked_scores: Same shape as scores
    """
```

### attention_weights
```python
def attention_weights(
    scores: np.ndarray,
    axis: int = -1
) -> np.ndarray:
    """
    Convert scores to attention weights via softmax.

    Args:
        scores: Attention scores
        axis: Axis to apply softmax (default: last axis)

    Returns:
        weights: Same shape as scores, sums to 1 along axis
    """
```

### scaled_dot_product_attention
```python
def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Complete scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Args:
        Q: Queries, shape (..., seq_q, d_k)
        K: Keys, shape (..., seq_k, d_k)
        V: Values, shape (..., seq_k, d_v)
        mask: Optional mask, shape broadcastable to (seq_q, seq_k)

    Returns:
        output: Attended values, shape (..., seq_q, d_v)
        weights: Attention weights, shape (..., seq_q, seq_k)
    """
```

### additive_attention
```python
def additive_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bahdanau-style additive attention.

    score(q, k) = v^T tanh(W_q @ q + W_k @ k)

    Args:
        Q: Queries, shape (batch, seq_q, d_q)
        K: Keys, shape (batch, seq_k, d_k)
        V: Values, shape (batch, seq_k, d_v)
        W_q: Query projection, shape (d_q, d_attn)
        W_k: Key projection, shape (d_k, d_attn)
        v: Score vector, shape (d_attn,)
        mask: Optional mask

    Returns:
        output: Attended values
        weights: Attention weights
    """
```

---

## Module: multi_head.py

### split_heads
```python
def split_heads(
    x: np.ndarray,
    num_heads: int
) -> np.ndarray:
    """
    Split last dimension into multiple heads.

    Args:
        x: Input, shape (batch, seq, d_model)
        num_heads: Number of attention heads

    Returns:
        split: Shape (batch, num_heads, seq, d_k)
               where d_k = d_model // num_heads

    Example:
        >>> x = np.random.randn(2, 10, 512)  # batch=2, seq=10, d_model=512
        >>> split = split_heads(x, num_heads=8)
        >>> split.shape
        (2, 8, 10, 64)  # 512/8 = 64
    """
```

### merge_heads
```python
def merge_heads(x: np.ndarray) -> np.ndarray:
    """
    Merge heads back into single dimension.

    Args:
        x: Input, shape (batch, num_heads, seq, d_k)

    Returns:
        merged: Shape (batch, seq, d_model)
                where d_model = num_heads * d_k
    """
```

### multi_head_attention_forward
```python
def multi_head_attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    W_O: np.ndarray,
    num_heads: int,
    mask: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """
    Multi-head attention forward pass.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
    where head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)

    Args:
        Q: Queries, shape (batch, seq_q, d_model)
        K: Keys, shape (batch, seq_k, d_model)
        V: Values, shape (batch, seq_k, d_model)
        W_Q: Query projection, shape (d_model, d_model)
        W_K: Key projection, shape (d_model, d_model)
        W_V: Value projection, shape (d_model, d_model)
        W_O: Output projection, shape (d_model, d_model)
        num_heads: Number of attention heads
        mask: Optional attention mask

    Returns:
        output: Shape (batch, seq_q, d_model)
        cache: Values for backward pass
    """
```

### multi_head_attention_backward
```python
def multi_head_attention_backward(
    grad_output: np.ndarray,
    cache: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Multi-head attention backward pass.

    Args:
        grad_output: Gradient from next layer
        cache: Cache from forward pass

    Returns:
        grad_Q: Gradient w.r.t. Q input
        grad_K: Gradient w.r.t. K input
        grad_V: Gradient w.r.t. V input
        grad_params: Dict with gradients for W_Q, W_K, W_V, W_O
    """
```

### MultiHeadAttention (class)
```python
class MultiHeadAttention:
    """
    Multi-head attention layer.

    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_k: Dimension per head (d_model // num_heads)
        W_Q, W_K, W_V, W_O: Projection matrices

    Methods:
        forward(Q, K, V, mask=None): Compute multi-head attention
        backward(grad_output): Compute gradients
        get_params(): Return parameters
        set_params(params): Set parameters
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
```

---

## Module: positional.py

### sinusoidal_encoding
```python
def sinusoidal_encoding(
    max_length: int,
    d_model: int
) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_length: Maximum sequence length
        d_model: Model dimension

    Returns:
        encodings: Shape (max_length, d_model)

    Example:
        >>> pe = sinusoidal_encoding(100, 512)
        >>> pe.shape
        (100, 512)
    """
```

### add_positional_encoding
```python
def add_positional_encoding(
    x: np.ndarray,
    pe: np.ndarray
) -> np.ndarray:
    """
    Add positional encoding to input embeddings.

    Args:
        x: Input embeddings, shape (batch, seq_len, d_model)
        pe: Positional encodings, shape (max_len, d_model)

    Returns:
        output: x + pe[:seq_len], same shape as x
    """
```

### learned_positional_encoding
```python
def learned_positional_encoding(
    max_length: int,
    d_model: int
) -> np.ndarray:
    """
    Initialize learned positional embeddings.

    Args:
        max_length: Maximum sequence length
        d_model: Model dimension

    Returns:
        embeddings: Shape (max_length, d_model), initialized randomly
    """
```

### create_causal_mask
```python
def create_causal_mask(seq_length: int) -> np.ndarray:
    """
    Create causal (look-ahead) mask for autoregressive attention.

    Args:
        seq_length: Sequence length

    Returns:
        mask: Shape (seq_length, seq_length)
              Lower triangular matrix (True below/on diagonal)

    Example:
        >>> mask = create_causal_mask(4)
        >>> mask
        array([[ True, False, False, False],
               [ True,  True, False, False],
               [ True,  True,  True, False],
               [ True,  True,  True,  True]])
    """
```

### create_padding_mask
```python
def create_padding_mask(
    lengths: np.ndarray,
    max_length: int
) -> np.ndarray:
    """
    Create padding mask from sequence lengths.

    Args:
        lengths: Actual sequence lengths, shape (batch,)
        max_length: Maximum sequence length

    Returns:
        mask: Shape (batch, max_length)
              True for real tokens, False for padding

    Example:
        >>> mask = create_padding_mask(np.array([3, 2]), max_length=4)
        >>> mask
        array([[ True,  True,  True, False],
               [ True,  True, False, False]])
    """
```

---

## Module: transformer_block.py

### layer_norm
```python
def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Layer normalization.

    y = gamma * (x - mean) / sqrt(var + eps) + beta

    Args:
        x: Input, shape (..., d_model)
        gamma: Scale parameter, shape (d_model,)
        beta: Shift parameter, shape (d_model,)
        eps: Small constant for numerical stability

    Returns:
        normalized: Same shape as x
    """
```

### layer_norm_backward
```python
def layer_norm_backward(
    grad_output: np.ndarray,
    x: np.ndarray,
    gamma: np.ndarray,
    eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass for layer normalization.

    Args:
        grad_output: Gradient from next layer
        x: Original input
        gamma: Scale parameter
        eps: Epsilon used in forward

    Returns:
        grad_x: Gradient w.r.t. input
        grad_gamma: Gradient w.r.t. gamma
        grad_beta: Gradient w.r.t. beta
    """
```

### feed_forward
```python
def feed_forward(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray
) -> np.ndarray:
    """
    Position-wise feed-forward network.

    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

    Args:
        x: Input, shape (batch, seq, d_model)
        W1: First layer weights, shape (d_model, d_ff)
        b1: First layer bias, shape (d_ff,)
        W2: Second layer weights, shape (d_ff, d_model)
        b2: Second layer bias, shape (d_model,)

    Returns:
        output: Shape (batch, seq, d_model)
    """
```

### TransformerEncoderBlock (class)
```python
class TransformerEncoderBlock:
    """
    Single Transformer encoder block.

    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> + -> LayerNorm -> FFN -> +
        |__________________________|           |___________________|

    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        attention: MultiHeadAttention layer
        W1, b1, W2, b2: FFN parameters
        gamma1, beta1, gamma2, beta2: LayerNorm parameters

    Methods:
        forward(x, mask=None): Process input
        backward(grad_output): Compute gradients
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None
    ):
        """
        Initialize encoder block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: FFN hidden dimension (default: 4 * d_model)
        """
```

### stack_encoder_blocks
```python
def stack_encoder_blocks(
    x: np.ndarray,
    blocks: list[TransformerEncoderBlock],
    mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Pass input through stack of encoder blocks.

    Args:
        x: Input embeddings, shape (batch, seq, d_model)
        blocks: List of TransformerEncoderBlock instances
        mask: Optional attention mask

    Returns:
        output: Encoded representations, shape (batch, seq, d_model)
    """
```
