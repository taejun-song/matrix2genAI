# Stage 02: Linear Algebra - Vectors and Matrices

**Duration**: 2-3 days
**Prerequisites**: Basic Python, NumPy basics
**Difficulty**: ⭐⭐☆☆☆

**The Big Idea:** Vectors and matrices are the language of machine learning. Understanding them deeply is the difference between using ML as a black box and truly mastering it.

## Getting Started

### Setup

```bash
# Navigate to this stage
cd stages/s02_linear_algebra

# Run tests (using uv - recommended)
uv run pytest tests/ -v

# Or activate venv first
source .venv/bin/activate  # On Unix/macOS
pytest tests/ -v
```

### Files You'll Edit

- `starter/vector_ops.py` - Dot product, norms, vector operations
- `starter/matrix_ops.py` - Matrix multiplication, transpose, trace
- `starter/linear_systems.py` - Solve Ax=b, Gaussian elimination
- `starter/decomposition.py` - LU, QR, eigenvalues

### Quick Test Commands

```bash
# Test specific module
uv run pytest tests/test_vector_ops.py -v

# Test specific function
uv run pytest tests/test_matrix_ops.py::TestMatrixMultiply -v

# Run all tests
uv run pytest tests/ -v
```

---

## What You'll Learn

Linear algebra is the mathematical foundation of machine learning. In this stage, you'll implement core operations from scratch to deeply understand:

- How vectors and matrices work at a fundamental level
- Why certain algorithms are numerically stable (and others aren't)
- The computational complexity of matrix operations
- Decompositions used throughout ML (LU, QR, eigenvalues)

## Conceptual Understanding

### Why Vectors and Matrices?

**Intuition:** Machine learning is about finding patterns in high-dimensional data. Vectors and matrices let us express this concisely.

```
Traditional programming:
  if user_age > 25 and user_income > 50000 and user_city == "NYC":
      recommend_product_A()

Machine learning:
  score = w₁·age + w₂·income + w₃·city_nyc + ... + w₁₀₀·feature₁₀₀

  In vector form:  score = w · x  (one line!)

  For 1000 users:  scores = W @ X  (still one line!)
```

### Vectors: Direction and Magnitude

```
Vector v = [3, 4]

Geometric view:
  • Direction: Northeast (from origin)
  • Magnitude: |v| = √(3² + 4²) = 5

ML view:
  • v could represent a data point (age=3, income=4)
  • |v| measures "size" or "importance"
  • v/|v| is the normalized direction (unit vector)
```

### Matrices: Transformations

```
Matrix A transforms vectors:

  A = [2  0]    v = [1]    Av = [2]
      [0  3]        [1]         [3]

  A stretches:
    • x-direction by 2×
    • y-direction by 3×

ML application:
  • Input x → Hidden layer: h = W₁x + b₁
  • W₁ transforms input space to hidden space
  • Every neural network layer is a matrix multiplication!
```

### The Dot Product: Measuring Similarity

**Intuition:** How aligned are two vectors?

```
Example: Recommendation system
  user_preferences = [5, 1, 0]  (loves action, dislikes romance, neutral on comedy)
  movie_features   = [4, 0, 2]  (action movie with some comedy)

  similarity = user · movie = 5×4 + 1×0 + 0×2 = 20

Compare to:
  romance_movie = [0, 5, 0]
  similarity = 5×0 + 1×5 + 0×0 = 5  (much less aligned!)

Key insight: Dot product is HIGH when vectors point in similar directions
```

**Geometric meaning:**
```
v · w = |v| |w| cos(θ)

  θ = 0°   → cos(θ) = 1  → v·w is maximum (parallel)
  θ = 90°  → cos(θ) = 0  → v·w = 0 (perpendicular)
  θ = 180° → cos(θ) = -1 → v·w is negative (opposite)
```

### Linear Systems: The Heart of ML

**Question:** Given m equations and n unknowns, can we solve for x?

```
Example: Fitting a line to 3 points

  Points: (1, 3), (2, 5), (3, 7)
  Model: y = mx + b

  System of equations:
    m·1 + b = 3
    m·2 + b = 5
    m·3 + b = 7

  Matrix form: Ax = y
    [1  1]   [m]   [3]
    [2  1] · [b] = [5]
    [3  1]         [7]

Solution: m=2, b=1  → y = 2x + 1 (perfect fit!)
```

**Why this matters:**
- **Linear regression**: Solve Ax = y for weights
- **Neural networks**: Each layer solves Wx + b = output
- **Least squares**: When exact solution doesn't exist, find best approximation

### Matrix Decomposition: Breaking Down Complexity

**Why decompose matrices?**

Think of factoring numbers: 24 = 2³ × 3
- Reveals structure (24 is even, divisible by 3)
- Makes computation easier (multiply small factors)

Matrices work the same way!

**LU Decomposition: A = LU**
```
Matrix A = Lower triangular × Upper triangular

Why useful?
  • Solving Ax = b becomes two easy steps:
    1. Solve Ly = b  (forward substitution)
    2. Solve Ux = y  (backward substitution)
  • Much faster than Gaussian elimination for multiple right-hand sides!

Application: Solving many linear systems with same A but different b
```

**QR Decomposition: A = QR**
```
Matrix A = Orthogonal matrix × Upper triangular

Q properties:
  • Columns are perpendicular (orthogonal)
  • Each column has length 1 (normalized)
  • Q·Qᵀ = I (perfect inverse)

Why useful?
  • Numerical stability for least squares
  • Basis for eigenvalue algorithms
  • Used in ML optimization algorithms

Application: Robust linear regression, PCA computation
```

**Eigenvalues/Eigenvectors: Special Directions**
```
Eigenvector v: A vector that doesn't change direction under A

  Av = λv

Example:
  A = [2  0]    v = [1]    Av = [2] = 2·[1]
      [0  3]        [0]         [0]     [0]

  v is eigenvector with eigenvalue λ = 2

Why this matters:
  • Principal Component Analysis (PCA): Find directions of maximum variance
  • PageRank: Eigenvector of web link matrix
  • Markov chains: Long-term behavior
  • Neural network analysis: Gradient flow, stability
```

## Why This Matters for ML

- **Neural networks**: Forward/backward passes are matrix multiplications
- **Linear regression**: Solved using linear systems (Ax = b)
- **PCA**: Uses eigenvalue decomposition
- **SVD**: Foundation of recommender systems, NLP
- **Optimization**: Gradients are vectors; Hessians are matrices

## Structure

### Part 1: Vector Operations (30 min)
Warm-up exercises on vector math.

### Part 2: Matrix Operations (1 hour)
Implement matrix multiplication two ways - naive and vectorized.

### Part 3: Linear Systems (2-3 hours)
Solve Ax = b using Gaussian elimination and LU decomposition.

### Part 4: Matrix Decomposition (3-4 hours)
Advanced: QR decomposition, power iteration for eigenvalues, determinants.

## Getting Started

1. Read [spec.md](spec.md) for detailed requirements
2. Implement functions in `starter/*.py`
3. Run tests frequently: `pytest .`
4. Grade yourself: `python ../../scripts/grade.py .`

## Tips

- **Start simple**: Get Part 1 working before moving to harder parts
- **Read the tests**: They show expected behavior and edge cases
- **NumPy docs**: Use `np.sum`, `np.dot`, broadcasting - just not `np.linalg.*` solvers
- **Numerical stability**: Use partial pivoting in Gaussian elimination
- **Debug**: Print intermediate results to understand algorithms

## Common Pitfalls

1. **Dimension mismatches**: Always check matrix/vector shapes before operating
2. **In-place modifications**: Copy arrays when needed (`A.copy()`)
3. **Integer division**: Use `float` types to avoid precision loss
4. **Zero division**: Handle zero vectors/singular matrices gracefully
5. **Off-by-one errors**: Carefully track indices in loops

## Extensions (Optional)

After getting 100%, try:
- Implement Cholesky decomposition for positive definite matrices
- Add support for sparse matrices
- Optimize matrix multiply with blocking for cache efficiency
- Implement inverse iteration for finding specific eigenvalues

## Resources

- [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [MIT OCW 18.06 - Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- [Numerical Linear Algebra](https://people.maths.ox.ac.uk/~trefethen/text.html) - Trefethen & Bau

## Next Stage

Once you achieve 100%, move on to **s03: Calculus - Derivatives and Gradients** where you'll use these linear algebra tools to compute gradients!
