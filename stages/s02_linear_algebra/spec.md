# Stage 02: Linear Algebra - Vectors and Matrices

## Overview

Implement fundamental linear algebra operations from scratch. These form the foundation of all machine learning algorithms.

## Mathematical Background

### Vectors
- A vector **v** ∈ ℝⁿ is an n-dimensional array of real numbers
- Operations: addition, scalar multiplication, dot product, norm

### Matrices
- A matrix **A** ∈ ℝᵐˣⁿ is an m×n array of real numbers
- Operations: addition, scalar multiplication, matrix multiplication, transpose

### Key Properties
- Dot product: **v** · **w** = Σᵢ vᵢwᵢ
- Euclidean norm: ‖**v**‖₂ = √(Σᵢ vᵢ²)
- Matrix multiplication: **(AB)ᵢⱼ** = Σₖ AᵢₖBₖⱼ
- Determinant: det(**A**) measures volume scaling
- Eigenvalues: **Av** = λ**v** for eigenvector **v** and eigenvalue λ

## Tasks

### Part 1: Vector Operations

Implement the following functions in `starter/vector_ops.py`:

1. **`vector_add(v: np.ndarray, w: np.ndarray) -> np.ndarray`**
   - Element-wise addition of two vectors
   - Must check dimensions match

2. **`vector_scalar_multiply(v: np.ndarray, c: float) -> np.ndarray`**
   - Multiply vector by scalar

3. **`dot_product(v: np.ndarray, w: np.ndarray) -> float`**
   - Compute dot product v · w
   - Use NumPy operations (no explicit loops)

4. **`vector_norm(v: np.ndarray, p: float = 2.0) -> float`**
   - Compute Lₚ norm: ‖v‖ₚ = (Σᵢ |vᵢ|ᵖ)^(1/p)
   - Support p=1 (Manhattan), p=2 (Euclidean), p=∞ (max norm)

5. **`cosine_similarity(v: np.ndarray, w: np.ndarray) -> float`**
   - Compute cos(θ) = (v·w) / (‖v‖‖w‖)
   - Handle zero vectors (return 0.0)

### Part 2: Matrix Operations

Implement the following functions in `starter/matrix_ops.py`:

1. **`matrix_multiply_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray`**
   - Naive O(n³) matrix multiplication using explicit loops
   - Check dimensions are compatible (A: m×k, B: k×n → result: m×n)

2. **`matrix_multiply_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray`**
   - Vectorized matrix multiplication using NumPy broadcasting
   - Should be faster than naive version

3. **`matrix_transpose(A: np.ndarray) -> np.ndarray`**
   - Return Aᵀ

4. **`matrix_trace(A: np.ndarray) -> float`**
   - Compute trace: tr(A) = Σᵢ Aᵢᵢ
   - Must be square matrix

5. **`frobenius_norm(A: np.ndarray) -> float`**
   - Compute ‖A‖_F = √(Σᵢⱼ Aᵢⱼ²)

### Part 3: Linear Systems

Implement the following functions in `starter/linear_systems.py`:

1. **`gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray`**
   - Solve Ax = b using Gaussian elimination with partial pivoting
   - Return solution vector x
   - Raise ValueError if system is singular

2. **`lu_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]`**
   - Decompose A = LU where L is lower triangular, U is upper triangular
   - Return (L, U)
   - Use Doolittle algorithm

3. **`forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray`**
   - Solve Lx = b for lower triangular L

4. **`backward_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray`**
   - Solve Ux = b for upper triangular U

### Part 4: Matrix Decomposition

Implement the following functions in `starter/decomposition.py`:

1. **`qr_decomposition_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]`**
   - Decompose A = QR using Gram-Schmidt process
   - Q is orthogonal (QᵀQ = I), R is upper triangular
   - Return (Q, R)

2. **`power_iteration(A: np.ndarray, num_iters: int = 100, tol: float = 1e-6) -> tuple[float, np.ndarray]`**
   - Find dominant eigenvalue and eigenvector using power iteration
   - Return (eigenvalue, eigenvector)
   - Eigenvector should be normalized (‖v‖ = 1)

3. **`determinant_recursive(A: np.ndarray) -> float`**
   - Compute determinant using cofactor expansion (recursive)
   - Only for small matrices (n ≤ 4)

4. **`determinant_lu(A: np.ndarray) -> float`**
   - Compute determinant using LU decomposition
   - det(A) = det(L) × det(U) = product of diagonal elements of U

## Constraints

- Use NumPy for array operations, but implement algorithms yourself
- Do NOT use `np.linalg.solve`, `np.linalg.qr`, `np.linalg.eig`, etc.
- You CAN use basic NumPy operations: indexing, slicing, broadcasting, `np.sum`, `np.dot`, `np.abs`, `np.max`, etc.
- Handle edge cases (empty matrices, dimension mismatches, singular matrices)
- Maintain numerical stability (use partial pivoting, normalize vectors)

## Numerical Considerations

1. **Stability**: Use partial pivoting in Gaussian elimination
2. **Precision**: Be aware of floating-point errors (use `np.allclose` for comparisons)
3. **Efficiency**: Vectorized operations are much faster than loops

## Expected Performance

- Matrix multiply (naive): O(n³)
- Matrix multiply (vectorized): Still O(n³) but with better constants
- Gaussian elimination: O(n³)
- Power iteration: O(kn²) where k is number of iterations

## Testing

Run tests with:
```bash
pytest stages/s02_linear_algebra/
```

Tests verify:
- Correctness vs NumPy reference implementations
- Numerical stability (condition number effects)
- Edge cases (singular matrices, zero vectors, dimension mismatches)
- Performance hints (vectorized vs naive)

## Learning Goals

After completing this stage, you should understand:
- How matrix multiplication works at a low level
- Why numerical stability matters (pivoting, normalization)
- The difference between O(n³) algorithms and their practical performance
- Fundamental decompositions (LU, QR) used throughout ML
- How eigenvalues are computed iteratively
