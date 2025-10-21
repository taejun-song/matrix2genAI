# Matrix2GenAI

**Learn AI from scratch**: A comprehensive, test-driven curriculum from mathematical foundations to generative models.

Inspired by Nand2Tetris, this curriculum teaches modern AI/ML by having you implement everything from first principles. No solutions provided - only problems and tests.

## Quick Start

```bash
# Install dependencies (requires uv)
uv sync

# Navigate to first stage
cd stages/s01_number_systems_and_bits

# Read the problem specification
cat spec.md

# Implement your solution in starter/*.py
# Run tests
pytest .

# Grade your work
python ../../scripts/grade.py .
```

## Curriculum Structure

**28 stages** organized into 6 parts:

1. **Mathematical Foundations** (s01-s05): Number systems, linear algebra, calculus, probability, optimization
2. **Machine Learning Basics** (s06-s10): Regression, classification, optimizers, regularization, decision trees
3. **Neural Networks & Deep Learning** (s11-s15): MLPs, CNNs, RNNs, attention
4. **Advanced Deep Learning** (s16-s19): Transformers, training techniques, embeddings, seq2seq
5. **Reinforcement Learning** (s20-s23): MDPs, Q-learning, policy gradients, actor-critic
6. **Generative AI** (s24-s28): VAEs, GANs, diffusion models, LLMs, RLHF

See [CURRICULUM.md](CURRICULUM.md) for detailed breakdown.

## Learning Approach

Each stage follows the same pattern:

1. **Read** `spec.md` - understand the problem
2. **Implement** functions in `starter/*.py` - replace TODOs
3. **Test** with `pytest` - verify correctness
4. **Grade** with autograder - track progress
5. **Iterate** until 100% score
6. **Move on** to next stage

**No solutions provided.** You learn by solving problems yourself.

## Autograding System

Check your progress automatically:

```bash
# Grade current stage
python scripts/grade.py .

# Grade specific stage
python scripts/grade.py s01_number_systems_and_bits

# Get JSON output
python scripts/grade.py s01_number_systems_and_bits --json

# Custom timeout (default 300s)
python scripts/grade.py s01_number_systems_and_bits --timeout 600
```

The grader outputs:
- Score (0-100%)
- Tests passed/failed
- Execution time
- Helpful feedback

Aim for **100%** before moving to the next stage!

## Prerequisites

- **Python 3.11+**
- **Basic Python knowledge** (functions, classes, NumPy)
- **Math background**: High school algebra and calculus (we'll build from here)
- **Time commitment**: 3-6 months (2-3 stages/week)

## Development Setup

```bash
# Clone repo
git clone https://github.com/yourusername/matrix2genAI.git
cd matrix2genAI

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Run tests for a stage
pytest stages/s01_number_systems_and_bits/

# Run all tests (not recommended initially)
pytest stages/

# Type checking
mypy .

# Linting
ruff check .
```

## Repository Structure

```
matrix2genAI/
├── README.md              # This file
├── CURRICULUM.md          # Detailed curriculum
├── scripts/
│   └── grade.py           # Autograding script
├── stages/
│   ├── s01_number_systems_and_bits/
│   ├── s02_linear_algebra/
│   └── ...                # 28 total stages
├── pyproject.toml         # Dependencies (managed by uv)
└── pytest.ini             # Test configuration
```

## Stage Structure

Each stage contains:

```
stages/sXX_topic_name/
├── README.md              # Stage overview
├── spec.md                # Problem specification
├── starter/
│   ├── __init__.py
│   └── *.py               # Skeleton code with TODOs
└── tests/
    ├── __init__.py
    └── test_*.py          # Comprehensive tests
```

## Testing Philosophy

Tests are designed to:
- ✅ Verify correctness
- ✅ Check edge cases
- ✅ Ensure numerical stability
- ✅ Guide implementation (read test code for hints!)
- ✅ Provide immediate feedback

**Tip**: Read the test files! They contain valuable hints about expected behavior.

## Learning Path

**Recommended order**: Follow stages sequentially (s01 → s28)

**Checkpoints** (review before continuing):
- After s05: Math foundations solid?
- After s10: ML basics clear?
- After s15: Deep learning intuition?
- After s19: Advanced DL mastered?
- After s23: RL concepts understood?
- After s28: GenAI complete!

## Contributing

Students:
- ❌ Do NOT share solutions publicly
- ✅ Ask conceptual questions in discussions
- ✅ Report bugs/issues
- ✅ Suggest improvements

Educators/Contributors:
- ✅ Add new stages
- ✅ Improve tests
- ✅ Enhance documentation
- ✅ Review PRs

## FAQ

**Q: Can I use NumPy/PyTorch for implementation?**
A: Check each stage's spec.md. Early stages require pure Python/NumPy to learn fundamentals. Later stages may allow frameworks.

**Q: Tests are failing. What do I do?**
A: Read the test code carefully - it shows expected behavior. Check error messages for hints. Use `pytest -v` for verbose output.

**Q: Can I skip stages?**
A: Not recommended. Each stage builds on previous ones. You'll struggle later if you skip fundamentals.

**Q: How long does this take?**
A: 3-6 months at 2-3 stages/week. Varies by background. Some stages take 2 days, others 5-6 days.

**Q: Is this enough to get an ML job?**
A: This gives you deep foundations. Combine with practical projects, papers, and domain knowledge for job readiness.

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

Inspired by:
- [Nand2Tetris](https://www.nand2tetris.org/) - Build a computer from scratch
- [fast.ai](https://www.fast.ai/) - Practical deep learning
- [CS231n](http://cs231n.stanford.edu/) - Stanford CNN course
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - OpenAI RL resource

---

**Ready to start?** Head to [stages/s01_number_systems_and_bits](stages/s01_number_systems_and_bits/) and begin your journey from fundamentals to generative AI!
