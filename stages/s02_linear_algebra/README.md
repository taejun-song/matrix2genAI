# Stage 02: Linear Algebra - Vectors and Matrices

**Duration**: 2-3 days
**Prerequisites**: Basic Python, NumPy basics
**Difficulty**: ⭐⭐☆☆☆

## What You'll Learn

Linear algebra is the mathematical foundation of machine learning. In this stage, you'll implement core operations from scratch to deeply understand:

- How vectors and matrices work at a fundamental level
- Why certain algorithms are numerically stable (and others aren't)
- The computational complexity of matrix operations
- Decompositions used throughout ML (LU, QR, eigenvalues)

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
