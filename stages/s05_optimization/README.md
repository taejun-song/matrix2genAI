# Stage 05: Optimization Fundamentals

**Duration**: 3-4 days
**Prerequisites**: s02, s03, s04
**Difficulty**: ⭐⭐⭐☆☆

**The Big Idea:** Optimization is the engine of machine learning. Without good optimizers, even the best model architecture won't learn. Understanding optimization means understanding how learning actually happens.

## Getting Started

### Setup

```bash
# Navigate to this stage
cd stages/s05_optimization

# Run tests (using uv - recommended)
uv run pytest tests/ -v

# Or activate venv first
source .venv/bin/activate  # On Unix/macOS
pytest tests/ -v
```

### Files You'll Edit

- `starter/gradient_descent.py` - Basic GD, SGD, mini-batch GD
- `starter/momentum.py` - Momentum, Nesterov momentum
- `starter/adaptive.py` - AdaGrad, RMSProp, Adam
- `starter/schedules.py` - Learning rate decay strategies

### Quick Test Commands

```bash
# Test specific module
uv run pytest tests/test_gradient_descent.py -v

# Test specific function
uv run pytest tests/test_adaptive.py::TestAdam -v

# Run all tests
uv run pytest tests/ -v
```

---

## What You'll Learn

Optimization is how we train ML models. You'll implement:
- Gradient descent and variants (momentum)
- Modern optimizers (AdaGrad, RMSProp, Adam)
- Learning rate schedules

## Conceptual Understanding

### The Optimization Problem

**Question:** How do we find the best model parameters?

```
We have:
  • Model: f(x; θ) with parameters θ
  • Loss function: L(θ) measuring how bad our predictions are
  • Goal: Find θ* that minimizes L(θ)

Example: Linear regression
  θ = [w, b]  (weights and bias)
  L(θ) = MSE = (1/n) Σ(y - (wx + b))²

  Find w* and b* that minimize MSE

Challenge: Can't try all possible values!
  • θ might have millions of dimensions
  • Need an efficient search strategy

Solution: Gradient descent!
```

### Gradient Descent: Following the Slope Downhill

**Intuition:** You're on a foggy mountain and want to reach the valley.

```
Strategy:
  1. Feel the slope under your feet (compute gradient)
  2. Take a step downhill (opposite direction of gradient)
  3. Repeat until you reach the bottom

Mathematical update:
  θ_{t+1} = θ_t - α · ∇L(θ_t)

  • θ_t: Current parameters
  • α: Learning rate (step size)
  • ∇L(θ_t): Gradient (direction of steepest increase)
  • Minus sign: Go opposite direction (downhill!)
```

**Why it works:**
```
Taylor approximation:
  L(θ + Δθ) ≈ L(θ) + ∇L(θ)ᵀ · Δθ

If we set Δθ = -α · ∇L(θ):
  L(θ - α·∇L(θ)) ≈ L(θ) - α · ||∇L(θ)||²

Since ||∇L(θ)||² ≥ 0, loss decreases!
(for small enough α)

Key insight: Gradient points toward increase, so -gradient points toward decrease
```

### Batch vs Stochastic vs Mini-Batch

**Batch Gradient Descent:**
```
Use entire dataset to compute gradient

Gradient: ∇L(θ) = (1/n) Σᵢ₌₁ⁿ ∇L_i(θ)

✓ Accurate gradient direction
✓ Smooth convergence
✗ Slow for large datasets (n=1,000,000 → compute 1M gradients per step!)
✗ Can't do online learning
✗ Gets stuck in local minima

Example: n = 1,000,000 images
  One gradient computation = process all 1M images
  Very expensive!
```

**Stochastic Gradient Descent (SGD):**
```
Use ONE random sample to estimate gradient

Gradient estimate: ∇L(θ) ≈ ∇L_i(θ)  (just one sample i)

✓ Fast updates (one sample per step)
✓ Can escape local minima (noisy updates)
✓ Enables online learning
✗ Noisy gradients → erratic path
✗ May not converge to exact minimum

Example: n = 1,000,000 images
  One gradient computation = process 1 image
  1,000,000× faster per update!
```

**Mini-Batch Gradient Descent (the goldilocks):**
```
Use small batch to estimate gradient

Gradient estimate: ∇L(θ) ≈ (1/B) Σᵢ∈batch ∇L_i(θ)
  where B = batch size (e.g., 32, 64, 256)

✓ Fast updates (much less than full dataset)
✓ Stable gradients (averaging reduces noise)
✓ Efficient on GPUs (parallel computation)
✓ Good generalization (some noise helps!)
✗ Need to tune batch size

This is what everyone uses in practice!

Example: n = 1,000,000, B = 256
  One gradient computation = process 256 images
  One epoch = 1,000,000 / 256 ≈ 3,906 updates
  Great balance of speed and stability!
```

### The Learning Rate: Most Important Hyperparameter

**Too small:**
```
α = 0.0001

Updates: θ_{t+1} = θ_t - 0.0001 · ∇L

Result:
  • Tiny steps
  • Very slow convergence
  • May take forever to reach minimum
  • Gets stuck on plateaus

Loss curve: 100 → 99.9 → 99.8 → 99.7 → ... (barely moving!)
```

**Too large:**
```
α = 10.0

Updates: θ_{t+1} = θ_t - 10.0 · ∇L

Result:
  • Huge steps
  • Overshoots minimum
  • Oscillates wildly
  • May diverge to infinity!

Loss curve: 100 → 50 → 200 → 25 → 500 → NaN (explodes!)
```

**Just right:**
```
α = 0.01

Updates: θ_{t+1} = θ_t - 0.01 · ∇L

Result:
  • Steady progress
  • Converges smoothly
  • Reaches minimum

Loss curve: 100 → 50 → 25 → 12 → 6 → 3 → 1.5 → 1.0 (converges!)

Finding the right α: Art + science
  • Try values: {0.001, 0.003, 0.01, 0.03, 0.1}
  • Use learning rate schedules (decay over time)
  • Use adaptive methods (Adam automatically adjusts)
```

### Momentum: Accelerating in Consistent Directions

**Problem with vanilla GD:**
```
Imagine a valley that's steep in one direction, flat in another:

  Loss landscape (contour plot):
    ║     ║  (steep vertically)
    ║  ·  ║  (· = current position)
    ║     ║
    ════════  (flat horizontally)

  Gradient descent zigzags:
    → ↓ → ↓ → ↓ → ↓ (slow horizontal progress!)

  Why? Large gradient vertically causes oscillation
       Small gradient horizontally → slow progress
```

**Solution: Momentum**
```
Idea: Accumulate velocity like a rolling ball

Update rule:
  v_{t+1} = β·v_t + ∇L(θ_t)     (velocity update)
  θ_{t+1} = θ_t - α·v_{t+1}     (parameter update)

  • v: velocity (exponential moving average of gradients)
  • β: momentum coefficient (typically 0.9)

Effect:
  • Consistent directions → build up speed
  • Oscillating directions → cancel out
  • Smooths out noisy gradients

Physical analogy:
  Ball rolling downhill accumulates momentum
  Goes faster in consistent downhill direction
  Friction (1-β) prevents infinite acceleration
```

**With momentum:**
```
Same loss landscape:
  ║     ║
  ║  ·  ║
  ║     ║
  ════════

Path with momentum:
  → → → → → → → (smooth, fast horizontal progress!)

  • Vertical oscillations damped
  • Horizontal momentum builds up
  • Much faster convergence
```

**Nesterov Momentum (look-ahead):**
```
Standard momentum: Compute gradient at current position
Nesterov: Compute gradient at "lookahead" position

Update rule:
  θ_lookahead = θ_t - β·v_t          (where we're about to go)
  v_{t+1} = β·v_t + ∇L(θ_lookahead)  (gradient at lookahead)
  θ_{t+1} = θ_t - α·v_{t+1}

Benefit: Better when approaching minimum
  • Can "correct" before overshooting
  • Slightly better convergence in practice
```

### Adaptive Learning Rates: Per-Parameter Adjustment

**Problem:** Same learning rate for all parameters might not be optimal

```
Consider two features:
  Feature 1: Ranges from 0 to 1
  Feature 2: Ranges from 0 to 1000

With α = 0.01:
  • Feature 1 weight: Good progress
  • Feature 2 weight: Too large (overshoots)

We want: Different learning rates for different parameters!
```

**AdaGrad: Adapt based on historical gradients**
```
Idea: Larger updates for infrequent parameters, smaller for frequent

Update rule:
  G_{t+1} = G_t + (∇L_t)²           (accumulate squared gradients)
  θ_{t+1} = θ_t - α / √(G_{t+1} + ε) · ∇L_t

Effect:
  • Parameters with large past gradients → small learning rate
  • Parameters with small past gradients → large learning rate
  • Automatically balances different scales

✓ Great for sparse data (NLP, recommender systems)
✗ Learning rate only decreases (G always grows)
✗ May stop learning too early
```

**RMSProp: Fix AdaGrad's aggressive decay**
```
Idea: Use moving average instead of sum

Update rule:
  G_{t+1} = β·G_t + (1-β)·(∇L_t)²   (exponential moving average)
  θ_{t+1} = θ_t - α / √(G_{t+1} + ε) · ∇L_t

Difference from AdaGrad:
  • Recent gradients matter more (exponential weighting)
  • G doesn't grow forever (old gradients forgotten)
  • Learning rate can increase or decrease

✓ Works well for non-stationary problems
✓ Used successfully in RNNs
```

**Adam: Combining Momentum + RMSProp**
```
The king of optimizers (most widely used)

Idea: Use both first moment (momentum) and second moment (RMSProp)

Update rule:
  m_{t+1} = β₁·m_t + (1-β₁)·∇L_t           (first moment - momentum)
  v_{t+1} = β₂·v_t + (1-β₂)·(∇L_t)²        (second moment - RMSProp)

  m̂ = m_{t+1} / (1 - β₁^{t+1})             (bias correction)
  v̂ = v_{t+1} / (1 - β₂^{t+1})             (bias correction)

  θ_{t+1} = θ_t - α · m̂ / (√v̂ + ε)

Parameters (typical values):
  • β₁ = 0.9   (momentum decay)
  • β₂ = 0.999 (RMSProp decay)
  • α = 0.001  (learning rate)
  • ε = 1e-8   (numerical stability)

Why bias correction?
  • m and v initialized to 0
  • Early estimates biased toward 0
  • Correction term fixes this

✓ Robust to hyperparameters (default values work well!)
✓ Fast convergence
✓ Works across many problems
✓ Memory efficient

This is the default choice for most deep learning!
```

### Learning Rate Schedules: Decay Over Time

**Intuition:** Start with large steps (explore), end with small steps (refine)

**Step Decay:**
```
Reduce α by factor every N epochs

Example: α₀ = 0.1, decay = 0.5, every 10 epochs
  Epochs 0-9:   α = 0.1
  Epochs 10-19: α = 0.05
  Epochs 20-29: α = 0.025
  ...

✓ Simple
✓ Works well in practice
✗ Need to tune decay schedule
```

**Exponential Decay:**
```
α_t = α₀ · decay^t

Example: α₀ = 0.1, decay = 0.96
  Epoch 0:  α = 0.1
  Epoch 10: α = 0.1 · 0.96^10 ≈ 0.066
  Epoch 50: α = 0.1 · 0.96^50 ≈ 0.013

✓ Smooth decay
✓ Continuous rather than discrete
```

**Cosine Annealing:**
```
α_t = α_min + (α_max - α_min) · (1 + cos(πt/T)) / 2

Effect: Smoothly decrease from α_max to α_min over T steps

Example: α_max = 0.1, α_min = 0.001, T = 100
  [Creates smooth cosine curve from 0.1 down to 0.001]

✓ Smooth, no discrete jumps
✓ Popular in modern deep learning
✓ Can restart (warm restarts)
```

**Why schedules help:**
```
Early training:
  • Large α → fast exploration
  • Find good region quickly

Late training:
  • Small α → fine-tuning
  • Converge to precise minimum

Without schedule:
  • Large α throughout → oscillates near minimum
  • Small α throughout → slow convergence
```

### Convergence: When to Stop?

**Criteria:**

1. **Loss stops improving:**
```
if |L_t - L_{t-1}| < threshold:
    stop

Example: threshold = 1e-6
  Epoch 98: Loss = 0.1234567
  Epoch 99: Loss = 0.1234568
  Change = 1e-7 < 1e-6 → converged!
```

2. **Gradient becomes tiny:**
```
if ||∇L_t|| < threshold:
    stop

At minimum: ∇L = 0
Near minimum: ∇L ≈ 0
```

3. **Validation loss increases (early stopping):**
```
Training loss keeps decreasing, but validation loss increases
→ Model is overfitting → STOP!

Best practice:
  • Track validation loss
  • Stop when it hasn't improved for N epochs
  • Restore best model (from lowest validation loss)
```

4. **Maximum iterations:**
```
Always set a max to prevent infinite loops!

if epoch >= max_epochs:
    stop
```

## Why This Matters

- **Training neural networks**: All use these optimizers
- **Understanding convergence**: Why some optimizers work better
- **Hyperparameter tuning**: Choosing learning rates, momentum, etc.
- **Debugging training**: Recognize when optimization is failing

## Next Stage

**s06: Linear Regression from Scratch** - Apply optimization to real ML!
