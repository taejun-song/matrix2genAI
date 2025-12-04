# Stage 15: Attention Mechanisms

## Overview

Attention mechanisms allow neural networks to focus on relevant parts of the input when producing each output. Originally developed for sequence-to-sequence models, attention became the foundation for Transformers, which revolutionized natural language processing and beyond.

## Learning Objectives

By completing this stage, you will:
- Understand attention as a soft addressing mechanism
- Implement scaled dot-product attention from scratch
- Build multi-head attention for parallel attention computation
- Create positional encodings for sequence position information
- Construct a complete Transformer encoder block

## Prerequisites

- Stage 14: Recurrent Neural Networks (sequence modeling concepts)
- Stage 12: Feedforward Networks (dense layers, backpropagation)
- Linear algebra (matrix multiplication, softmax)

## Conceptual Foundation

### Why Attention?

RNNs process sequences step-by-step, creating a bottleneck: all information must pass through a fixed-size hidden state. For long sequences, early information gets "compressed" or lost.

**The Problem with RNNs:**
```
Input:  [word1, word2, word3, ..., word100]
                    ↓
         Sequential processing
                    ↓
Output: Single hidden state must remember everything
```

**Attention Solution:**
```
For each output position:
  - Look at ALL input positions
  - Compute relevance scores
  - Weight inputs by relevance
  - Combine weighted inputs
```

### The Attention Intuition

Think of attention like a database query:
- **Query (Q)**: What am I looking for?
- **Key (K)**: What does each item contain?
- **Value (V)**: What information to retrieve?

The attention score between a query and key tells us how much to weight the corresponding value.

### Scaled Dot-Product Attention

The core attention mechanism:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Step by step:
1. **Compute scores**: `QK^T` - dot product between query and all keys
2. **Scale**: Divide by `√d_k` to prevent softmax saturation
3. **Softmax**: Convert scores to probabilities (weights sum to 1)
4. **Weighted sum**: Multiply weights by values

**Example:**
```
Q = [1, 0]  (looking for something)
K = [[1, 0],   (matches query)
     [0, 1],   (doesn't match)
     [0.5, 0.5]]  (partial match)

Scores = Q @ K.T = [1, 0, 0.5]
Weights = softmax(scores) ≈ [0.5, 0.2, 0.3]  (approximately)

Output = weighted sum of V rows
```

### Why Scale by √d_k?

For large `d_k`, dot products can become very large, pushing softmax into regions with tiny gradients.

If `q` and `k` have variance 1, then `q·k` has variance `d_k`. Dividing by `√d_k` normalizes the variance back to 1.

### Self-Attention

In self-attention, Q, K, and V all come from the same sequence:

```python
Q = X @ W_Q  # Project input to queries
K = X @ W_K  # Project input to keys
V = X @ W_V  # Project input to values
output = Attention(Q, K, V)
```

Each position can attend to all positions (including itself), learning which parts of the sequence are relevant to each other.

### Attention Masks

Sometimes we need to prevent attention to certain positions:

**Padding Mask**: Ignore padding tokens in variable-length sequences
```
mask = [[1, 1, 1, 0, 0],  # real, real, real, pad, pad
        [1, 1, 0, 0, 0]]  # real, real, pad, pad, pad
scores = scores + (1 - mask) * (-1e9)  # Large negative → 0 after softmax
```

**Causal Mask**: Prevent attending to future positions (for autoregressive models)
```
[[1, 0, 0, 0],   Position 0 sees only itself
 [1, 1, 0, 0],   Position 1 sees 0, 1
 [1, 1, 1, 0],   Position 2 sees 0, 1, 2
 [1, 1, 1, 1]]   Position 3 sees all
```

## Multi-Head Attention

Instead of one attention function, we run several in parallel:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

where head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)
```

**Why multiple heads?**
- Different heads can focus on different aspects (syntax, semantics, etc.)
- More expressive than single attention
- Each head operates on a subspace (d_model / h dimensions)

**Dimensions:**
- Input: `d_model`
- Per-head: `d_k = d_v = d_model / h`
- Output: `d_model` (after concatenation and projection)

### Head Splitting and Merging

```python
# Split: (batch, seq, d_model) → (batch, heads, seq, d_k)
def split_heads(x, num_heads):
    batch, seq, d_model = x.shape
    d_k = d_model // num_heads
    x = x.reshape(batch, seq, num_heads, d_k)
    return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)

# Merge: (batch, heads, seq, d_k) → (batch, seq, d_model)
def merge_heads(x):
    batch, heads, seq, d_k = x.shape
    x = x.transpose(0, 2, 1, 3)  # (batch, seq, heads, d_k)
    return x.reshape(batch, seq, heads * d_k)
```

## Positional Encoding

Attention is permutation-invariant—it doesn't inherently know position. We must add position information explicitly.

### Sinusoidal Encoding

The original Transformer uses fixed sinusoidal patterns:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Properties:**
- Each position has a unique encoding
- Relative positions can be computed via linear transformation
- Extrapolates to longer sequences than seen during training

**Why sin/cos?**
- Bounded values (-1 to 1)
- Different frequencies capture different scales of position
- `PE(pos+k)` can be expressed as linear function of `PE(pos)`

### Learned Positional Encoding

Alternative: Learn position embeddings as parameters
```python
pos_embedding = nn.Embedding(max_length, d_model)
```

Works well when max sequence length is known and fixed.

## Transformer Architecture Components

### Layer Normalization

Normalizes across features (not batch):
```python
def layer_norm(x, gamma, beta, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta
```

Unlike batch norm, works with any batch size (including 1).

### Position-wise Feed-Forward Network

Two linear layers with activation:
```python
def feed_forward(x, W1, b1, W2, b2):
    hidden = relu(x @ W1 + b1)  # Expand to d_ff
    return hidden @ W2 + b2      # Back to d_model
```

Typically `d_ff = 4 * d_model`.

### Residual Connections

Add input to output of sublayer:
```python
output = layer_norm(x + sublayer(x))
```

Or with "Pre-LN" (more stable training):
```python
output = x + sublayer(layer_norm(x))
```

### Complete Encoder Block

```python
def encoder_block(x):
    # Self-attention sublayer
    attn_out = multi_head_attention(x, x, x)  # Q=K=V=x
    x = layer_norm(x + attn_out)

    # Feed-forward sublayer
    ff_out = feed_forward(x)
    x = layer_norm(x + ff_out)

    return x
```

## Implementation Tips

### Efficient Batched Attention

Process entire batch with matrix operations:
```python
# scores: (batch, heads, seq_q, seq_k)
scores = (Q @ K.transpose(-2, -1)) / sqrt(d_k)
weights = softmax(scores, axis=-1)
output = weights @ V  # (batch, heads, seq_q, d_v)
```

### Numerical Stability

For softmax:
```python
def stable_softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)
```

For attention with masking:
```python
if mask is not None:
    scores = scores + (1 - mask) * (-1e9)  # Before softmax
```

### Gradient Flow

Attention has good gradient flow:
- No sequential dependencies (unlike RNN)
- Direct paths from any input to any output
- Residual connections help gradients flow

## Testing Your Implementation

Run the tests:
```bash
cd stages/s15_attention
uv run pytest tests/ -v
```

### Test Categories

1. **Attention Core**: Score computation, softmax, weighted sum
2. **Multi-Head**: Head splitting, merging, full forward pass
3. **Positional**: Sinusoidal encoding values and shapes
4. **Transformer Block**: Full encoder block forward/backward

## Success Criteria

- [ ] `compute_attention_scores` produces correct QK^T / √d_k
- [ ] `attention_weights` applies softmax correctly
- [ ] `scaled_dot_product_attention` combines scores and values
- [ ] `apply_attention_mask` zeros out masked positions
- [ ] `split_heads` correctly reshapes for multi-head
- [ ] `merge_heads` reverses split_heads
- [ ] `multi_head_attention_forward` produces correct shapes
- [ ] `sinusoidal_encoding` follows the formula
- [ ] `layer_norm` normalizes along feature dimension
- [ ] `TransformerEncoderBlock` processes sequences correctly

## Common Mistakes

1. **Wrong transpose**: K must be transposed, not Q
2. **Forgetting scaling**: Division by √d_k is crucial
3. **Mask dimensions**: Mask shape must broadcast correctly
4. **Head dimensions**: Ensure d_model is divisible by num_heads
5. **Softmax axis**: Must be along the key dimension

## Mathematical Reference

### Attention Equations

**Single-head:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Multi-head:**
```
MultiHead(Q, K, V) = [head_1; ...; head_h] W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Positional Encoding

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

### Layer Normalization

```
y = γ * (x - μ) / √(σ² + ε) + β
where μ, σ computed over last dimension
```

## Historical Context

- **2014**: Attention for machine translation (Bahdanau et al.)
- **2015**: "Attention is All You Need" idea developing
- **2017**: Transformer architecture (Vaswani et al.)
- **2018**: BERT, GPT-1
- **2019+**: GPT-2, GPT-3, T5, and explosion of transformer models
- **2022+**: ChatGPT, Claude, and modern LLMs

## The Road to GPT

The Transformer encoder block you've built is the foundation of:
- **BERT**: Stack of encoder blocks (bidirectional)
- **GPT**: Stack of decoder blocks (causal attention)
- **T5**: Encoder-decoder architecture

Modern LLMs are essentially very deep stacks of these blocks with:
- More layers (GPT-3: 96 layers)
- Larger dimensions (d_model up to 12288)
- Optimizations (sparse attention, flash attention, etc.)

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238)

## Congratulations!

You've now implemented the core building blocks from basic matrix operations to attention mechanisms—the foundation of modern AI! From here, you could explore:
- Decoder blocks and autoregressive generation
- Full encoder-decoder architectures
- Vision Transformers (ViT)
- Mixture of Experts
