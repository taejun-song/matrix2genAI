# Stage 14: Recurrent Neural Networks

## Overview

Recurrent Neural Networks (RNNs) process sequential data by maintaining a hidden state that captures information from previous time steps. Unlike feedforward networks that process inputs independently, RNNs can model temporal dependencies in sequences like text, speech, and time series.

## Learning Objectives

By completing this stage, you will:
- Understand how hidden states enable sequence modeling
- Implement vanilla RNN forward and backward passes
- Build LSTM cells to capture long-term dependencies
- Implement GRU as a simplified gating mechanism
- Apply gradient clipping to prevent exploding gradients

## Prerequisites

- Stage 12: Feedforward Networks (backpropagation)
- Stage 11: Activation functions (tanh, sigmoid)
- Matrix calculus (chain rule through time)

## Conceptual Foundation

### The Sequence Modeling Problem

Many real-world problems involve sequential data:
- **Language**: Words depend on previous words
- **Time series**: Stock prices depend on history
- **Audio**: Sound samples form temporal patterns

Feedforward networks treat each input independently:
```
input_1 → output_1
input_2 → output_2
input_3 → output_3  (no connection between them)
```

RNNs connect them through hidden state:
```
input_1 → h_1 → output_1
              ↘
input_2 → h_2 → output_2
              ↘
input_3 → h_3 → output_3
```

### Vanilla RNN

The simplest RNN computes:

```
h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
y_t = W_hy · h_t + b_y
```

Where:
- `x_t`: Input at time t
- `h_t`: Hidden state at time t
- `y_t`: Output at time t
- `W_xh`: Input-to-hidden weights
- `W_hh`: Hidden-to-hidden weights (the "recurrent" connection)
- `W_hy`: Hidden-to-output weights

**The Key Insight**: `W_hh` is applied at every time step, allowing information to flow from past to present.

### Unrolling Through Time

To understand RNN computation, "unroll" it:

```
Time:    t=0        t=1        t=2
         ↓          ↓          ↓
Input:   x_0        x_1        x_2
         ↓          ↓          ↓
Hidden: h_0 ----→ h_1 ----→ h_2
         ↓          ↓          ↓
Output:  y_0        y_1        y_2
```

Each horizontal arrow uses the same `W_hh` matrix.

### Backpropagation Through Time (BPTT)

To train RNNs, we backpropagate gradients through the unrolled network:

1. **Forward pass**: Compute all hidden states and outputs
2. **Backward pass**: Gradients flow back through time

The gradient of the loss w.r.t. `h_t` depends on:
- Direct gradient from `y_t`
- Gradient flowing back from `h_{t+1}` through `W_hh`

```python
# Simplified BPTT
for t in reversed(range(T)):
    grad_h[t] = grad_output[t] + grad_h[t+1] @ W_hh.T  # Key: gradient flows back!
    grad_W_hh += grad_h[t] @ h[t-1].T
```

### The Vanishing Gradient Problem

When backpropagating through many time steps:

```
∂L/∂h_0 = ∂L/∂h_T · ∂h_T/∂h_{T-1} · ... · ∂h_1/∂h_0
```

Each `∂h_t/∂h_{t-1}` involves multiplying by `W_hh` and the tanh derivative.

- If `||W_hh|| < 1`: Gradients shrink exponentially → **Vanishing gradients**
- If `||W_hh|| > 1`: Gradients grow exponentially → **Exploding gradients**

This limits vanilla RNNs to learning short-term dependencies (~10-20 steps).

### Gradient Clipping

For exploding gradients, we clip:

```python
def clip_gradients(gradients, max_norm):
    total_norm = sqrt(sum(g.norm()**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        gradients = [g * scale for g in gradients]
    return gradients
```

This prevents updates from being too large while preserving direction.

## LSTM: Long Short-Term Memory

LSTMs (Hochreiter & Schmidhuber, 1997) solve vanishing gradients with **gating mechanisms**.

### Cell State: The Memory Highway

LSTMs introduce a cell state `c_t` that flows through time with minimal transformation:

```
c_{t-1} ----[×]----[+]---→ c_t
             ↑      ↑
           forget  input
            gate    gate
```

The cell state is like a conveyor belt—information flows along it with additive updates, avoiding multiplicative degradation.

### The Three Gates

**1. Forget Gate** - What to forget from cell state:
```
f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)
```
Output in [0,1]. Values close to 0 = forget, close to 1 = keep.

**2. Input Gate** - What new information to store:
```
i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  # Candidate values
```
`i_t` decides how much of the candidate to add.

**3. Output Gate** - What to output:
```
o_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(c_t)
```

### LSTM Equations Summary

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     # Input gate
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  # Candidate
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     # Output gate

c_t = f_t * c_{t-1} + i_t * c̃_t         # Update cell state
h_t = o_t * tanh(c_t)                    # Update hidden state
```

### Why LSTMs Work

The key is the cell state update:
```
c_t = f_t * c_{t-1} + i_t * c̃_t
```

Gradient flows through addition, not multiplication by weight matrices. If `f_t ≈ 1`, gradients pass through unchanged, enabling learning over hundreds of time steps.

## GRU: Gated Recurrent Unit

GRUs (Cho et al., 2014) simplify LSTMs by combining gates:

### GRU Equations

```
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)     # Update gate
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)     # Reset gate
h̃_t = tanh(W_h · [r_t * h_{t-1}, x_t] + b_h)  # Candidate
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t  # Update hidden state
```

### GRU vs LSTM

| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| States | Cell + Hidden | Hidden only |
| Parameters | More | Fewer |
| Performance | Better for long sequences | Similar, faster training |

GRU merges the forget and input gates into a single "update gate" and combines the cell and hidden states.

## Implementation Patterns

### Processing Sequences

```python
# Forward pass through sequence
h = initial_hidden_state
outputs = []
for t in range(seq_length):
    h = rnn_cell(x[t], h)
    outputs.append(h)
```

### Batch Processing

Real implementations process batches:
- Input: `(batch_size, seq_length, input_size)`
- Hidden: `(batch_size, hidden_size)`
- Output: `(batch_size, seq_length, hidden_size)`

### Many-to-One vs Many-to-Many

**Many-to-One** (e.g., sentiment classification):
- Process entire sequence
- Use only final hidden state for prediction

**Many-to-Many** (e.g., language modeling):
- Process entire sequence
- Produce output at every time step

### Bidirectional RNNs

Process sequence in both directions:
```
Forward:  h_1 → h_2 → h_3
Backward: h_1 ← h_2 ← h_3
Output: concat(forward, backward)
```

## Testing Your Implementation

Run the tests:
```bash
cd stages/s14_rnn
uv run pytest tests/ -v
```

### Test Categories

1. **RNN Cell**: Single step forward/backward correctness
2. **Full RNN**: Sequence processing, hidden state shapes
3. **LSTM**: Gate computations, cell state updates
4. **GRU**: Gating mechanism, state updates
5. **Gradient Clipping**: Norm computation and scaling

## Success Criteria

- [ ] `rnn_cell_forward` produces correct hidden state shape
- [ ] `rnn_cell_backward` computes gradients for all weights
- [ ] `rnn_forward` processes full sequences correctly
- [ ] `lstm_cell_forward` implements all four gates
- [ ] `lstm_cell_backward` handles cell and hidden state gradients
- [ ] `gru_cell_forward` implements update and reset gates
- [ ] `clip_gradients` correctly scales when norm exceeds threshold
- [ ] Networks can learn simple sequence patterns

## Common Mistakes

1. **Wrong concatenation**: `[h, x]` concatenation dimension matters
2. **Forgetting initial states**: h_0 and c_0 must be initialized
3. **Gradient accumulation**: Gradients sum across time steps
4. **Gate confusion**: Careful with sigmoid vs tanh
5. **Shape mismatches**: Track (batch, hidden) vs (batch, seq, hidden)

## Mathematical Reference

### Vanilla RNN Gradients

For loss L at time T only:
```
∂L/∂W_hh = Σ_t (∂L/∂h_t · h_{t-1}.T)
∂L/∂h_t = ∂L/∂y_t · W_hy.T + ∂L/∂h_{t+1} · W_hh.T · diag(1 - h_{t+1}²)
```

### LSTM Gradients

The cell state gradient:
```
∂L/∂c_t = ∂L/∂h_t · o_t · (1 - tanh²(c_t)) + ∂L/∂c_{t+1} · f_{t+1}
```

Note: `f_{t+1}` term enables gradient flow through time.

## Practical Tips

### Initialization

- Initialize hidden state to zeros
- Initialize forget gate bias to 1-2 (encourages remembering early in training)
- Use orthogonal initialization for recurrent weights

### Training

- Use gradient clipping (max_norm = 1-5)
- Start with small sequences, increase length gradually
- LSTM/GRU > vanilla RNN for sequences > 20 steps
- Consider learning rate warmup

### Debugging

- Check hidden state doesn't explode (print norm)
- Verify gradient norms are reasonable
- Start with memorization task (learn to output previous input)

## Historical Context

- **1986**: Backpropagation through time (Rumelhart et al.)
- **1990**: Vanishing gradient problem identified
- **1997**: LSTM introduced (Hochreiter & Schmidhuber)
- **2014**: GRU introduced (Cho et al.)
- **2017**: Transformers begin replacing RNNs for many tasks

## Further Reading

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069)

## Next Steps

After completing this stage:
- **Stage 15**: Attention Mechanisms (the path to Transformers)
