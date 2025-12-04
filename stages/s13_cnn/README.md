# Stage 13: Convolutional Neural Networks

## Overview

Convolutional Neural Networks (CNNs) revolutionized computer vision by learning spatial hierarchies of features directly from raw pixel data. Unlike fully connected networks that treat input as a flat vector, CNNs preserve spatial structure through local connectivity and weight sharing.

## Learning Objectives

By completing this stage, you will:
- Understand convolution as a feature extraction operation
- Implement efficient convolution using im2col transformation
- Build pooling layers for spatial downsampling
- Create a complete CNN architecture (LeNet)

## Prerequisites

- Stage 12: Feedforward Networks (backpropagation, layers)
- Linear algebra (matrix operations)
- NumPy broadcasting

## Conceptual Foundation

### Why Convolutions?

Consider a 28×28 grayscale image. A fully connected layer would need 784 input weights per neuron. For 100 neurons, that's 78,400 parameters just for the first layer!

CNNs solve this through two key ideas:

1. **Local Connectivity**: Each neuron connects only to a small region (receptive field)
2. **Weight Sharing**: Same weights (filters) applied across all positions

A 3×3 filter has only 9 parameters but can detect features anywhere in the image.

### The Convolution Operation

Mathematically, 2D convolution is:

```
output[i,j] = Σ_m Σ_n input[i+m, j+n] × filter[m, n]
```

The filter "slides" across the input, computing dot products at each position.

**Example**: Edge detection with a 3×3 filter:
```
Input (5×5):           Filter (3×3):        Output (3×3):
1 1 1 0 0              1  0 -1             -4 -4 -4
1 1 1 0 0              1  0 -1              0  0  0
1 1 1 0 0       *      1  0 -1      =       4  4  4
0 0 1 1 1
0 0 1 1 1
```

The filter detects vertical edges (transitions from bright to dark).

### Padding

Without padding, convolution shrinks the output:
- Input: H × W
- Filter: k × k
- Output: (H - k + 1) × (W - k + 1)

**Types of padding**:
- **Valid (no padding)**: Output shrinks
- **Same**: Pad to keep output size equal to input
- **Full**: Pad so filter can be centered on every pixel

For "same" padding with a k×k filter: pad = k // 2

### Stride

Stride controls how far the filter moves between positions.

- **Stride 1**: Move one pixel at a time
- **Stride 2**: Move two pixels (halves output size)

Output size formula:
```
output_size = floor((input_size + 2*padding - filter_size) / stride) + 1
```

### Multiple Channels

Real images have multiple channels (RGB = 3 channels). A conv layer:
- Input: (H, W, C_in) - C_in input channels
- Filter: (k, k, C_in, C_out) - C_out filters, each with C_in channels
- Output: (H', W', C_out)

Each output channel is the sum of convolving across all input channels:
```
output[:,:,c_out] = Σ_c_in conv(input[:,:,c_in], filter[:,:,c_in,c_out])
```

### The im2col Trick

Naive convolution uses nested loops and is slow. The im2col transformation converts convolution to matrix multiplication:

1. **Extract patches**: Each k×k patch becomes a row
2. **Reshape filter**: Flatten to (k*k*C_in, C_out)
3. **Matrix multiply**: patches @ filters = output
4. **Reshape output**: Back to spatial format

This leverages highly optimized BLAS routines.

**Example**:
```
Input (4×4):              Patches (im2col):
1 2 3 4                   1 2 5 6
5 6 7 8       →           2 3 6 7
9 10 11 12                3 4 7 8
13 14 15 16               ...
```

### Backpropagation Through Convolution

The backward pass computes:
1. **Gradient w.r.t. input**: Full convolution with rotated filter
2. **Gradient w.r.t. filter**: Convolution of input with output gradient

Using im2col makes this straightforward:
```python
# Forward: output = im2col(input) @ filter
# Backward for input: grad_input = col2im(grad_output @ filter.T)
# Backward for filter: grad_filter = im2col(input).T @ grad_output
```

## Pooling Layers

### Purpose

Pooling provides:
- **Translation invariance**: Small shifts don't affect output
- **Dimensionality reduction**: Compress spatial dimensions
- **Larger receptive field**: Later layers see more of the image

### Max Pooling

Takes the maximum value in each pool region:
```
Input (4×4):              Max Pool (2×2, stride 2):
1 3 2 1                   6 4
5 6 4 2       →           9 8
8 9 5 3
7 4 8 7
```

**Backpropagation**: Gradient flows only to the max element. Store indices during forward pass.

### Average Pooling

Takes the average value in each pool region:
```
Input (4×4):              Avg Pool (2×2, stride 2):
1 3 2 1                   3.75 2.25
5 6 4 2       →           7.0  5.75
8 9 5 3
7 4 8 7
```

**Backpropagation**: Gradient is distributed equally to all elements in the pool.

### Global Average Pooling

Pools across entire spatial dimensions:
- Input: (H, W, C)
- Output: (1, 1, C) or just (C,)

Used in modern architectures to replace fully connected layers at the end.

## Classic Architecture: LeNet-5

Yann LeCun's LeNet-5 (1998) pioneered CNNs for digit recognition:

```
Input: 32×32×1 (grayscale)
    ↓
Conv1: 6 filters, 5×5 → 28×28×6
    ↓
Pool1: 2×2, stride 2 → 14×14×6
    ↓
Conv2: 16 filters, 5×5 → 10×10×16
    ↓
Pool2: 2×2, stride 2 → 5×5×16
    ↓
Flatten: 400
    ↓
FC1: 400 → 120 (ReLU)
    ↓
FC2: 120 → 84 (ReLU)
    ↓
Output: 84 → 10 (softmax)
```

This pattern—convolution followed by pooling, repeated, then fully connected—became the template for CNNs.

## Implementation Details

### Batch Processing

Modern implementations process batches:
- Input: (batch_size, H, W, C_in) or (batch_size, C_in, H, W)
- This stage uses NHWC format (channels last)

### Data Layout

Two common conventions:
- **NHWC**: (batch, height, width, channels) - TensorFlow default
- **NCHW**: (batch, channels, height, width) - PyTorch default

We use NHWC as it's more intuitive for beginners.

### Numerical Stability

- Initialize conv filters with He initialization (adjusted for fan_in = k*k*C_in)
- Use small learning rates (CNNs are sensitive to initialization)
- Consider batch normalization (not covered in this stage)

## Common Patterns

### Receptive Field Growth

Each conv layer increases the receptive field:
- After k×k conv: receptive field = k
- After another k×k conv: receptive field = 2k - 1
- Pooling with stride s multiplies effective receptive field by s

### Feature Hierarchy

Early layers learn simple features:
- Layer 1: Edges, gradients
- Layer 2: Textures, patterns
- Layer 3: Parts of objects
- Layer 4+: Whole objects, scenes

### Modern Additions (Beyond This Stage)

- **Batch Normalization**: Normalize activations for faster training
- **Skip Connections**: ResNet's identity shortcuts
- **Dilated Convolutions**: Larger receptive field without pooling
- **Depthwise Separable**: Efficient factorization (MobileNet)

## Testing Your Implementation

Run the tests:
```bash
cd stages/s13_cnn
uv run pytest tests/ -v
```

### Test Categories

1. **Convolution utilities**: Padding, im2col/col2im correctness
2. **Conv layer**: Forward/backward shapes, gradient checking
3. **Pooling**: Max/avg pooling forward/backward
4. **Integration**: Full CNN on simple patterns

## Success Criteria

- [ ] `pad2d` correctly pads arrays with specified values
- [ ] `im2col` extracts patches matching naive convolution
- [ ] `col2im` reconstructs from patches (inverse of im2col)
- [ ] `Conv2D.forward` produces correct output shapes
- [ ] `Conv2D.backward` passes gradient checking
- [ ] `MaxPool2D` returns correct max values and indices
- [ ] `AvgPool2D` computes correct averages
- [ ] LeNet architecture classifies simple patterns

## Common Mistakes

1. **Off-by-one errors**: Double-check output size formulas
2. **Wrong axis**: Summing over wrong dimension in channel aggregation
3. **Forgetting bias**: Conv layers usually have bias terms
4. **Index errors in pooling**: Careful with non-divisible dimensions
5. **Gradient shape mismatch**: Backward output must match forward input shape

## Mathematical Reference

### Forward Pass Equations

**Convolution**:
```
Z[n,h,w,c] = Σ_i Σ_j Σ_k X[n, h*s+i, w*s+j, k] × W[i,j,k,c] + b[c]
```

**Max Pooling**:
```
Y[n,h,w,c] = max_{i,j ∈ pool} X[n, h*s+i, w*s+j, c]
```

**Average Pooling**:
```
Y[n,h,w,c] = (1/pool_size²) Σ_{i,j ∈ pool} X[n, h*s+i, w*s+j, c]
```

### Backward Pass Equations

**Conv gradient w.r.t. input**:
```
dX[n,h,w,k] = Σ_c Σ_i Σ_j dZ[n,h',w',c] × W[i,j,k,c]
where h' = (h-i)/s, w' = (w-j)/s (if valid indices)
```

**Conv gradient w.r.t. weights**:
```
dW[i,j,k,c] = Σ_n Σ_h Σ_w dZ[n,h,w,c] × X[n, h*s+i, w*s+j, k]
```

## Further Reading

- [LeCun et al. "Gradient-Based Learning Applied to Document Recognition" (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)
- [An Introduction to Convolutional Neural Networks](https://arxiv.org/abs/1511.08458)

## Next Steps

After completing this stage, you'll be ready for:
- **Stage 14**: Recurrent Neural Networks (sequence modeling)
- **Stage 15**: Attention Mechanisms (transformers)
