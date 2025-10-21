# Matrix2GenAI Curriculum

A comprehensive, hands-on curriculum for learning AI from mathematical foundations to generative models.

## Philosophy

- **Learn by building**: Implement everything from scratch
- **Test-driven**: Each stage provides pytest tests; students write code to pass them
- **Progressive**: Each stage builds on previous concepts
- **No solutions provided**: Students must solve problems themselves
- **Autograded**: Automated grading system to track progress

## Curriculum Overview

### Part 1: Mathematical Foundations (s01-s05)

#### s01: Number Systems and Bits ✓
**Status**: Implemented
**Duration**: 2-3 days
**Prerequisites**: Basic Python, NumPy basics
**Concepts**: Fixed-point arithmetic, quantization, numerical stability
**Problems**:
- Fixed-point encoding/decoding
- Fixed-point arithmetic with overflow handling
- Quantization error bounds

**Learning Objectives**:
- Understand how numbers are represented in computers
- Recognize numerical stability issues in ML
- Implement low-level arithmetic operations

#### s02: Linear Algebra - Vectors and Matrices
**Duration**: 2-3 days
**Prerequisites**: Basic Python, NumPy basics
**Concepts**: Vector operations, matrix multiplication, determinants, inverses, eigenvalues
**Problems**:
- Implement vector dot product, norm, distance
- Matrix multiplication (naive and blocked)
- Gaussian elimination for solving linear systems
- Computing eigenvalues/eigenvectors (power iteration)
- Matrix decomposition (LU, QR basics)

**Learning Objectives**:
- Master fundamental linear algebra operations
- Understand computational complexity of matrix operations
- Build intuition for numerical stability

**Tests**:
- Correctness vs NumPy reference
- Numerical stability checks
- Edge cases (singular matrices, zero vectors)

#### s03: Calculus - Derivatives and Gradients
**Duration**: 2-3 days
**Prerequisites**: s02
**Concepts**: Numerical differentiation, partial derivatives, gradients, chain rule
**Problems**:
- Numerical gradient computation (finite differences)
- Implement automatic differentiation (forward mode)
- Implement reverse-mode AD (simple backprop)
- Gradient checking utilities
- Jacobian and Hessian computation

**Learning Objectives**:
- Understand how gradients are computed
- Build foundation for backpropagation
- Learn numerical differentiation techniques

**Tests**:
- Gradient correctness for polynomial, trig, exp functions
- Chain rule verification
- Numerical stability (step size)

#### s04: Probability and Statistics
**Duration**: 2-3 days
**Prerequisites**: s02, s03
**Concepts**: Distributions, expectation, variance, covariance, MLE, MAP
**Problems**:
- Implement common distributions (Normal, Bernoulli, Categorical)
- Sample generation and PDF/PMF computation
- Computing mean, variance, covariance from data
- Maximum Likelihood Estimation for simple distributions
- Bayesian inference (simple cases)

**Learning Objectives**:
- Understand probabilistic foundations of ML
- Implement statistical estimators
- Build intuition for uncertainty

**Tests**:
- Statistical properties (mean, variance of samples)
- MLE parameter recovery
- Bayes rule verification

#### s05: Optimization Fundamentals
**Duration**: 3-4 days
**Prerequisites**: s02, s03, s04
**Concepts**: Gradient descent, convexity, line search, convergence
**Problems**:
- Implement gradient descent with learning rate scheduling
- Implement Newton's method
- Line search methods (backtracking, exact)
- Constrained optimization (projected gradient descent)
- Convergence analysis tools

**Learning Objectives**:
- Master optimization algorithms used in ML
- Understand convergence properties
- Learn hyperparameter tuning for optimizers

**Tests**:
- Convergence on convex quadratic functions
- Finding minima of test functions (Rosenbrock, etc.)
- Learning rate sensitivity

---

### Part 2: Machine Learning Fundamentals (s06-s10)

#### s06: Linear Regression from Scratch
**Duration**: 2-3 days
**Prerequisites**: s02, s03, s05
**Concepts**: Least squares, normal equations, gradient descent, feature scaling
**Problems**:
- Implement linear regression (closed-form solution)
- Implement with gradient descent
- Feature normalization/standardization
- Polynomial feature expansion
- R² score and MSE computation

**Learning Objectives**:
- First complete ML algorithm implementation
- Understand optimization vs closed-form solutions
- Learn feature engineering basics

**Tests**:
- Convergence to analytical solution
- Prediction accuracy on toy datasets
- Feature scaling correctness

#### s07: Logistic Regression and Classification
**Duration**: 2-3 days
**Prerequisites**: s06
**Concepts**: Sigmoid, cross-entropy loss, binary/multiclass classification
**Problems**:
- Implement sigmoid and softmax functions
- Binary logistic regression with gradient descent
- Multiclass logistic regression (one-vs-all, softmax)
- Cross-entropy loss computation
- Accuracy, precision, recall, F1 metrics

**Learning Objectives**:
- Transition from regression to classification
- Understand loss functions for classification
- Learn evaluation metrics

**Tests**:
- Binary classification on synthetic data
- Multiclass classification (Iris dataset format)
- Metric computation correctness

#### s08: Gradient Descent Variants
**Duration**: 3-4 days
**Prerequisites**: s06, s07
**Concepts**: SGD, mini-batch, momentum, RMSprop, Adam
**Problems**:
- Implement batch, mini-batch, SGD
- Implement momentum-based optimizers
- Implement RMSprop
- Implement Adam optimizer
- Learning rate schedules (step decay, exponential)

**Learning Objectives**:
- Understand modern optimization algorithms
- Learn adaptive learning rate methods
- Compare optimization strategies

**Tests**:
- Convergence speed comparisons
- Optimization on non-convex functions
- Hyperparameter sensitivity

#### s09: Regularization and Overfitting
**Duration**: 2-3 days
**Prerequisites**: s06, s07
**Concepts**: L1/L2 regularization, cross-validation, bias-variance tradeoff
**Problems**:
- Implement L1 (Lasso) and L2 (Ridge) regularization
- K-fold cross-validation
- Train/validation/test split utilities
- Learning curves visualization data
- Early stopping implementation

**Learning Objectives**:
- Understand overfitting and generalization
- Learn model selection techniques
- Implement validation strategies

**Tests**:
- Regularization reduces overfitting
- CV selects better hyperparameters
- Early stopping prevents overfitting

#### s10: Decision Trees and Ensembles
**Duration**: 3-4 days
**Prerequisites**: s04, s07
**Concepts**: Information gain, Gini index, tree building, random forests
**Problems**:
- Implement decision tree (ID3/CART algorithm)
- Information gain and Gini impurity
- Tree pruning
- Bootstrap sampling
- Random forest (bagging + feature randomness)

**Learning Objectives**:
- Learn non-linear models
- Understand ensemble methods
- Implement tree-based algorithms

**Tests**:
- Tree correctly splits data
- Forest outperforms single tree
- Feature importance computation

---

### Part 3: Neural Networks and Deep Learning (s11-s15)

#### s11: Single Neuron and Activation Functions
**Duration**: 2 days
**Prerequisites**: s07, s08
**Concepts**: Perceptron, activations (ReLU, tanh, sigmoid), forward/backward pass
**Problems**:
- Implement activation functions (ReLU, LeakyReLU, tanh, sigmoid, Swish)
- Implement their derivatives
- Single neuron forward/backward pass
- Perceptron learning algorithm

**Learning Objectives**:
- Build foundation for neural networks
- Understand non-linear activations
- Implement gradient computation

**Tests**:
- Activation function shapes and ranges
- Gradient correctness (numerical vs analytical)
- Perceptron learns linearly separable data

#### s12: Feedforward Neural Networks (Backpropagation)
**Duration**: 4-5 days
**Prerequisites**: s11
**Concepts**: Multi-layer perceptron, backpropagation, weight initialization
**Problems**:
- Implement dense layer (forward/backward)
- Implement multi-layer network
- Backpropagation algorithm
- Weight initialization strategies (Xavier, He)
- Mini-batch training loop

**Learning Objectives**:
- Master backpropagation algorithm
- Understand deep learning fundamentals
- Learn initialization techniques

**Tests**:
- Gradient checking for backprop
- Network learns XOR problem
- Network learns MNIST-like data
- Different initializations affect training

#### s13: Convolutional Neural Networks
**Duration**: 4-5 days
**Prerequisites**: s12
**Concepts**: Convolution, pooling, CNNs for images, receptive fields
**Problems**:
- Implement 2D convolution (forward/backward)
- Implement max/average pooling
- Implement convolutional layer
- Build simple CNN architecture
- Flatten layer for transitioning to dense

**Learning Objectives**:
- Understand CNNs for computer vision
- Learn spatial feature extraction
- Implement efficient convolution

**Tests**:
- Convolution output shapes correct
- Gradient checking for conv layer
- CNN learns simple image classification
- Pooling reduces dimensions correctly

#### s14: Recurrent Neural Networks
**Duration**: 4-5 days
**Prerequisites**: s12
**Concepts**: RNN, LSTM, GRU, sequence modeling, BPTT
**Problems**:
- Implement vanilla RNN cell (forward/backward)
- Implement LSTM cell
- Implement GRU cell
- Backpropagation through time (BPTT)
- Sequence-to-sequence wrapper

**Learning Objectives**:
- Understand sequence modeling
- Learn gating mechanisms
- Master BPTT algorithm

**Tests**:
- RNN learns simple sequences (echo, counting)
- LSTM handles longer dependencies
- Gradient flow in LSTM vs vanilla RNN
- Sequence generation

#### s15: Attention Mechanisms
**Duration**: 3-4 days
**Prerequisites**: s14
**Concepts**: Attention, self-attention, multi-head attention, additive vs dot-product
**Problems**:
- Implement scaled dot-product attention
- Implement multi-head attention
- Implement additive (Bahdanau) attention
- Positional encoding
- Attention mask utilities

**Learning Objectives**:
- Understand attention for sequence modeling
- Learn self-attention mechanism
- Build foundation for Transformers

**Tests**:
- Attention weights sum to 1
- Attention focuses on relevant positions
- Multi-head captures different patterns
- Masked attention prevents future leakage

---

### Part 4: Advanced Deep Learning (s16-s19)

#### s16: Transformer Architecture
**Duration**: 5-6 days
**Prerequisites**: s15
**Concepts**: Transformer encoder/decoder, layer norm, residual connections
**Problems**:
- Implement layer normalization
- Implement feed-forward block
- Implement transformer encoder block
- Implement transformer decoder block
- Complete transformer model

**Learning Objectives**:
- Master Transformer architecture
- Understand modern NLP foundation
- Learn residual connections and normalization

**Tests**:
- Encoder-decoder dimensions match
- Causal masking in decoder
- Translation toy task
- Model learns sequence transformations

#### s17: Training Techniques
**Duration**: 3-4 days
**Prerequisites**: s12, s13
**Concepts**: BatchNorm, Dropout, initialization, gradient clipping
**Problems**:
- Implement batch normalization (train/eval modes)
- Implement dropout
- Implement gradient clipping
- Implement various initialization schemes
- Learning rate warmup

**Learning Objectives**:
- Learn advanced training techniques
- Understand regularization in deep learning
- Master training stabilization methods

**Tests**:
- BatchNorm stabilizes training
- Dropout prevents overfitting
- Gradient clipping prevents explosion
- Proper train/eval mode switching

#### s18: Embeddings and Word Vectors
**Duration**: 3-4 days
**Prerequisites**: s12
**Concepts**: Word embeddings, Word2Vec (Skip-gram, CBOW), negative sampling
**Problems**:
- Implement embedding layer
- Implement Skip-gram model
- Implement CBOW model
- Negative sampling
- Embedding similarity and analogies

**Learning Objectives**:
- Understand word representations
- Learn unsupervised learning for NLP
- Implement efficient training techniques

**Tests**:
- Embeddings capture semantic similarity
- Analogies work (king - man + woman ≈ queen)
- Negative sampling improves efficiency

#### s19: Sequence-to-Sequence Models
**Duration**: 3-4 days
**Prerequisites**: s14, s15
**Concepts**: Encoder-decoder, teacher forcing, beam search
**Problems**:
- Implement encoder-decoder architecture
- Implement teacher forcing
- Implement beam search
- Implement greedy decoding
- Attention-based seq2seq

**Learning Objectives**:
- Understand seq2seq modeling
- Learn decoding strategies
- Build translation systems

**Tests**:
- Model learns reversing sequences
- Model learns simple translation
- Beam search improves quality
- Teacher forcing vs free-running

---

### Part 5: Reinforcement Learning (s20-s23)

#### s20: Markov Decision Processes
**Duration**: 3-4 days
**Prerequisites**: s04
**Concepts**: States, actions, rewards, policies, value functions
**Problems**:
- Implement MDP environment (GridWorld)
- Policy evaluation (iterative)
- Value iteration
- Policy iteration
- Bellman equations

**Learning Objectives**:
- Understand RL fundamentals
- Learn dynamic programming
- Implement MDP solvers

**Tests**:
- Value iteration converges
- Optimal policy found for GridWorld
- Bellman consistency checks

#### s21: Q-Learning and SARSA
**Duration**: 3-4 days
**Prerequisites**: s20
**Concepts**: Temporal difference learning, Q-learning, SARSA, exploration vs exploitation
**Problems**:
- Implement tabular Q-learning
- Implement SARSA
- ε-greedy exploration
- Q-function approximation (simple)
- Experience replay buffer

**Learning Objectives**:
- Learn model-free RL
- Understand on-policy vs off-policy
- Implement value-based methods

**Tests**:
- Q-learning converges to optimal policy
- SARSA vs Q-learning (on-policy vs off-policy)
- Exploration rate decay
- Agent solves FrozenLake-like environment

#### s22: Policy Gradients
**Duration**: 4-5 days
**Prerequisites**: s12, s21
**Concepts**: REINFORCE, policy gradient theorem, baseline
**Problems**:
- Implement REINFORCE algorithm
- Implement baseline (value function)
- Monte Carlo returns computation
- Policy network (neural network policy)
- Advantage estimation

**Learning Objectives**:
- Understand policy-based methods
- Learn policy gradient theorem
- Implement variance reduction techniques

**Tests**:
- REINFORCE learns CartPole-like task
- Baseline reduces variance
- Policy network gradients correct

#### s23: Actor-Critic Methods
**Duration**: 4-5 days
**Prerequisites**: s22
**Concepts**: Actor-critic, A2C, PPO, trust regions
**Problems**:
- Implement advantage actor-critic (A2C)
- Implement generalized advantage estimation (GAE)
- Implement PPO clipping
- Implement value network training
- Parallel environment wrapper

**Learning Objectives**:
- Master modern RL algorithms
- Understand trust region methods
- Learn parallel training

**Tests**:
- A2C learns continuous control
- PPO improves over vanilla PG
- GAE reduces variance
- Clipping prevents destructive updates

---

### Part 6: Generative AI (s24-s28)

#### s24: Autoencoders
**Duration**: 3-4 days
**Prerequisites**: s12, s13
**Concepts**: Autoencoders, VAE, latent space, reparameterization trick
**Problems**:
- Implement vanilla autoencoder
- Implement variational autoencoder (VAE)
- Reparameterization trick
- KL divergence computation
- Latent space interpolation

**Learning Objectives**:
- Understand generative models
- Learn variational inference
- Implement VAE

**Tests**:
- AE reconstructs images
- VAE generates new samples
- KL divergence prevents collapse
- Latent space is smooth

#### s25: Generative Adversarial Networks
**Duration**: 4-5 days
**Prerequisites**: s12, s13
**Concepts**: GANs, generator, discriminator, adversarial training
**Problems**:
- Implement generator network
- Implement discriminator network
- GAN training loop (alternating optimization)
- Mode collapse detection
- Various GAN losses (WGAN, LSGAN)

**Learning Objectives**:
- Understand adversarial training
- Learn GAN training techniques
- Handle training instabilities

**Tests**:
- GAN generates realistic samples
- Training stabilizes
- Different losses affect quality
- FID/IS score computation

#### s26: Diffusion Models
**Duration**: 5-6 days
**Prerequisites**: s12, s13, s04
**Concepts**: DDPM, noise schedule, denoising, reverse diffusion
**Problems**:
- Implement forward diffusion process
- Implement noise schedule (linear, cosine)
- Implement denoising U-Net
- Implement reverse diffusion sampling
- DDIM sampling (faster)

**Learning Objectives**:
- Understand diffusion models
- Learn iterative refinement generation
- Implement modern image generation

**Tests**:
- Forward process adds noise correctly
- Denoising network learns to reverse
- Sampling generates images
- DDIM faster than DDPM

#### s27: Large Language Models Basics
**Duration**: 5-6 days
**Prerequisites**: s16, s18
**Concepts**: GPT architecture, causal language modeling, tokenization
**Problems**:
- Implement BPE tokenizer
- Implement GPT architecture (decoder-only transformer)
- Causal language modeling loss
- Text generation (sampling strategies)
- Temperature and top-k/top-p sampling

**Learning Objectives**:
- Understand LLM architecture
- Learn autoregressive generation
- Implement sampling strategies

**Tests**:
- Model learns next-token prediction
- Different sampling strategies vary output
- Temperature controls randomness
- Model generates coherent text

#### s28: Fine-tuning and RLHF
**Duration**: 5-6 days
**Prerequisites**: s22, s27
**Concepts**: Transfer learning, fine-tuning, RLHF, reward modeling
**Problems**:
- Implement supervised fine-tuning
- Implement reward model training
- Implement PPO for RLHF
- KL divergence constraint (vs reference model)
- Efficient fine-tuning (LoRA concept)

**Learning Objectives**:
- Understand transfer learning
- Learn alignment techniques
- Implement RLHF pipeline

**Tests**:
- Fine-tuning improves task performance
- Reward model learns preferences
- RLHF aligns model to reward
- KL penalty prevents drift

---

## Repository Structure

```
matrix2genAI/
├── README.md                 # Overview and getting started
├── CURRICULUM.md            # This file - complete curriculum
├── scripts/
│   └── grade.py             # Autograding script
├── stages/
│   ├── s01_number_systems_and_bits/
│   │   ├── spec.md          # Problem specification
│   │   ├── README.md        # Stage overview
│   │   ├── starter/
│   │   │   └── *.py         # Skeleton code with TODOs
│   │   └── tests/
│   │       └── test_*.py    # Pytest tests
│   ├── s02_linear_algebra/
│   │   └── ...
│   └── ...
├── pyproject.toml
└── pytest.ini
```

## Testing Philosophy

Each stage provides comprehensive tests:
1. **Correctness tests**: Verify implementation matches specifications
2. **Edge case tests**: Handle boundary conditions
3. **Numerical stability tests**: Check for numerical issues
4. **Performance hints**: Some tests hint at efficiency (but focus is correctness)
5. **Integration tests**: Ensure components work together

## Autograding

Students can grade their work using:

```bash
# Grade a specific stage
python scripts/grade.py s01_number_systems_and_bits

# Or using full path
python scripts/grade.py stages/s01_number_systems_and_bits

# Get JSON output
python scripts/grade.py s01_number_systems_and_bits --json
```

The autograder:
- Runs all tests for a stage
- Computes a score (0-100%)
- Provides detailed feedback
- Tracks test timing
- Supports timeout per test

## Progression Guidelines

- **Difficulty curve**: Gradual increase, with harder stages in Parts 4-6
- **Estimated time**: 3-6 months for complete curriculum (depends on background)
- **Checkpoints**: End of each Part (6 checkpoints total)
- **Prerequisites**: Each stage lists required prior stages
- **Recommended pace**: 2-3 stages per week

## Learning Outcomes

By completing this curriculum, students will:
- ✅ Understand AI/ML from first principles
- ✅ Implement core algorithms from scratch
- ✅ Debug and test numerical code
- ✅ Build production-quality implementations
- ✅ Understand modern generative AI (LLMs, diffusion models)
- ✅ Be prepared for research or industry roles in AI/ML

## Getting Started

1. Clone the repository
2. Install dependencies: `uv sync`
3. Start with s01: `cd stages/s01_number_systems_and_bits`
4. Read `spec.md` and `README.md`
5. Implement TODOs in `starter/*.py`
6. Test your work: `pytest .`
7. Grade yourself: `python ../../scripts/grade.py .`
8. Move to next stage when you get 100%!

## Contributing

This is an educational repository. Students should:
- ❌ NOT share solutions publicly
- ✅ Ask questions about concepts
- ✅ Discuss testing strategies
- ✅ Share learning experiences

Instructors can:
- ✅ Extend the curriculum
- ✅ Add new stages
- ✅ Improve tests
- ✅ Add documentation
