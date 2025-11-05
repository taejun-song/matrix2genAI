# Matrix to Generative AI: Complete Curriculum Roadmap

## Overview

A comprehensive, hands-on curriculum teaching the complete journey from mathematical foundations to state-of-the-art generative AI. Each stage builds on previous knowledge through implementation-focused assignments.

**Total Duration:** 15-20 weeks (3-4 stages per week)
**Target Audience:** Students with basic programming knowledge
**Learning Style:** Learn by building - implement everything from scratch

---

## Part 1: Mathematical Foundations (Weeks 1-2) ✅ COMPLETED

### s01: Number Systems and Bits (2-3 days) ⭐⭐
**Status:** ✅ Complete

**Topics:**
- Fixed-point arithmetic
- Quantization and numerical stability
- Overflow handling (saturation vs wrapping)

**Key Functions:** `to_fixed`, `from_fixed`, `fxp_add`, `fxp_mul`

**Why it matters:** Understanding numerical representations is crucial for ML optimization and model quantization.

---

### s02: Linear Algebra - Vectors and Matrices (2-3 days) ⭐⭐⭐
**Status:** ✅ Complete

**Topics:**
- Vector operations (add, dot product, norms)
- Matrix operations (multiply, transpose, trace)
- Linear systems (Gaussian elimination, LU decomposition)
- Matrix decompositions (QR, eigenvalues)

**Key Functions:** 18 functions across 4 modules

**Why it matters:** Linear algebra is the language of machine learning. Neural networks are matrix multiplication chains.

---

### s03: Calculus - Derivatives and Gradients (2-3 days) ⭐⭐⭐
**Status:** ✅ Complete

**Topics:**
- Numerical differentiation (finite differences)
- Automatic differentiation (forward mode)
- Reverse-mode autodiff (backpropagation foundation)
- Gradient checking

**Key Functions:** 8+ functions including computational graph

**Why it matters:** Backpropagation is reverse-mode autodiff. You'll implement the core of PyTorch/TensorFlow.

---

### s04: Probability and Statistics (2-3 days) ⭐⭐
**Status:** ✅ Complete

**Topics:**
- Probability distributions (Normal, Bernoulli, Categorical)
- Statistical measures (mean, variance, covariance)
- Maximum Likelihood Estimation (MLE)
- Maximum A Posteriori (MAP) estimation

**Key Functions:** 9 functions across 3 modules

**Why it matters:** ML is probabilistic modeling. Understanding distributions is essential for generative models.

---

### s05: Optimization Fundamentals (3-4 days) ⭐⭐⭐
**Status:** ✅ Complete

**Topics:**
- Gradient descent variants (vanilla, momentum)
- Adaptive optimizers (AdaGrad, RMSprop, Adam)
- Learning rate schedules
- Line search methods

**Key Functions:** 9 functions across 3 modules

**Why it matters:** Training neural networks is an optimization problem. You'll use these optimizers daily.

---

## Part 2: Machine Learning Fundamentals (Weeks 3-4)

### s06: Linear Regression & Gradient Descent (3-4 days) ⭐⭐⭐
**Status:** ✅ Complete

**Topics:**
- Linear models (univariate and multivariate)
- Normal equations (analytical solution)
- Gradient descent optimization
- Cost functions and metrics (MSE, MAE, R²)
- Feature scaling (standardization, normalization)
- Train/test splitting

**Key Implementations:**
- `LinearRegression` class (normal equations + gradient descent)
- 5 evaluation metrics
- `StandardScaler`, `MinMaxScaler`
- `train_test_split`, `polynomial_features`

**Prerequisites:** s02 (Linear Algebra), s03 (Calculus), s05 (Optimization)

**Why it matters:** First complete ML pipeline. Connects math to prediction. Foundation for all supervised learning.

---

### s07: Logistic Regression & Classification (3-4 days) ⭐⭐⭐
**Status:** ✅ Complete

**Topics:**
- Binary classification
- Sigmoid function and logistic model
- Cross-entropy loss
- Multi-class classification (softmax)
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Decision boundaries

**Key Implementations:**
- 15 functions across 3 modules (activations, losses, metrics)
- `sigmoid`, `softmax` activation functions
- Binary and multi-class cross-entropy loss and gradients
- Complete classification metrics suite
- Comprehensive 70-test suite

**Real-world applications:** Spam detection, medical diagnosis, fraud detection

---

### s08: Feature Engineering & Data Preprocessing (2-3 days) ⭐⭐
**Status:** ✅ Complete

**Topics:**
- Handling missing data (imputation strategies)
- Categorical encoding (one-hot, label encoding)
- Feature selection (variance threshold, correlation)
- Outlier detection (IQR method)
- Scaling and transformation (min-max, robust)

**Key Implementations:**
- 12 functions across 3 modules (imputation, encoding, scaling)
- Missing data handling (mean/median/mode/constant imputation)
- Label and one-hot encoding
- Min-max and robust scaling
- Variance threshold and correlation filtering
- Comprehensive 60-test suite

**Real-world applications:** Data cleaning pipelines, dimensionality reduction

---

### s09: Regularization & Model Selection (3-4 days) ⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Overfitting vs underfitting
- L1 regularization (Lasso) - feature selection
- L2 regularization (Ridge) - weight decay
- Elastic Net (combined L1+L2)
- Cross-validation (k-fold, stratified)
- Hyperparameter tuning
- Learning curves and validation curves

**Key Implementations:**
- `Ridge`, `Lasso`, `ElasticNet` regression
- `KFoldCV` class
- Grid search for hyperparameters
- Learning curve generators
- Regularization path visualization

**Real-world applications:** Preventing overfitting, model selection, automated ML

---

### s10: Decision Trees & Ensemble Methods (4-5 days) ⭐⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Decision tree learning (CART algorithm)
- Information gain and Gini impurity
- Tree pruning
- Random Forests (bagging)
- Gradient Boosting basics
- Feature importance from trees

**Key Implementations:**
- `DecisionTree` class (classification + regression)
- Split criteria (Gini, entropy, MSE)
- `RandomForest` with bootstrap sampling
- Simple gradient boosting
- Feature importance extraction

**Real-world applications:** Tabular data competitions, interpretable models

---

## Part 3: Neural Networks & Deep Learning (Weeks 5-7)

### s11: Perceptrons & Activation Functions (2-3 days) ⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Perceptron algorithm
- Activation functions (sigmoid, tanh, ReLU, Leaky ReLU, ELU, GELU)
- Forward propagation
- Single-layer networks
- XOR problem and non-linearity

**Key Implementations:**
- `Perceptron` class
- 8+ activation functions with derivatives
- Single-layer neural network
- Activation function comparison and visualization

**Prerequisites:** s03 (autodiff), s06-s07 (regression/classification)

**Why it matters:** Neural networks are compositions of these building blocks.

---

### s12: Feedforward Networks & Backpropagation (4-5 days) ⭐⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Multi-layer perceptrons (MLPs)
- Backpropagation algorithm (full derivation)
- Weight initialization strategies
- Vanishing/exploding gradients
- Forward and backward passes
- Mini-batch training

**Key Implementations:**
- `Layer` class (Dense/Fully Connected)
- `NeuralNetwork` class with arbitrary depth
- Complete backpropagation engine
- Multiple initialization methods (Xavier, He)
- Mini-batch gradient descent

**Real-world applications:** Image classification (MNIST), tabular prediction

---

### s13: Convolutional Neural Networks (4-5 days) ⭐⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Convolution operation (2D)
- Padding and stride
- Pooling layers (max, average)
- CNN architectures (LeNet-style)
- Feature maps and receptive fields
- Parameter sharing

**Key Implementations:**
- `Conv2D` layer with backprop
- `MaxPool2D`, `AvgPool2D` layers
- `Flatten` layer
- Simple CNN architecture
- Image preprocessing utilities

**Real-world applications:** Image classification, object detection foundations

---

### s14: Recurrent Neural Networks (4-5 days) ⭐⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Vanilla RNN architecture
- Backpropagation through time (BPTT)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Sequence-to-sequence basics
- Handling variable-length sequences

**Key Implementations:**
- `RNN` cell with forward/backward
- `LSTM` cell (forget, input, output gates)
- `GRU` cell
- Simple sequence models
- BPTT implementation

**Real-world applications:** Time series prediction, language modeling basics

---

### s15: Attention Mechanisms (3-4 days) ⭐⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Attention intuition (alignment)
- Scaled dot-product attention
- Multi-head attention
- Self-attention
- Positional encoding (sinusoidal)
- Attention visualization

**Key Implementations:**
- `ScaledDotProductAttention`
- `MultiHeadAttention` layer
- `PositionalEncoding`
- Attention weight visualization
- Simple attention-based sequence model

**Prerequisites:** s14 (RNNs), strong understanding of matrix operations

**Why it matters:** Foundation for Transformers and all modern LLMs.

---

## Part 4: Advanced Deep Learning (Weeks 8-9)

### s16: Transformers Architecture (5-6 days) ⭐⭐⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Transformer encoder architecture
- Transformer decoder architecture
- Feed-forward networks in Transformers
- Layer normalization
- Residual connections
- Masked attention
- Complete encoder-decoder model

**Key Implementations:**
- `TransformerEncoderLayer`
- `TransformerDecoderLayer`
- `LayerNorm`
- Full `Transformer` model
- Training loop for sequence tasks

**Real-world applications:** Machine translation, text generation, foundation for GPT/BERT

---

### s17: Training Techniques & Regularization (3-4 days) ⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Batch Normalization
- Dropout (standard and variants)
- Data augmentation (images and text)
- Gradient clipping
- Early stopping
- Model checkpointing
- Learning rate warmup

**Key Implementations:**
- `BatchNorm1D`, `BatchNorm2D`
- `Dropout` layer
- Image augmentation functions
- Training utilities (callbacks, checkpointing)
- Learning rate schedulers with warmup

**Real-world applications:** Training stable, generalizable models

---

### s18: Embeddings & Word Vectors (3-4 days) ⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Word embeddings concept
- Word2Vec (Skip-gram and CBOW)
- Negative sampling
- GloVe intuition
- Embedding layer implementation
- Token/vocabulary management
- Subword tokenization basics

**Key Implementations:**
- `Embedding` layer with gradient updates
- Word2Vec training (Skip-gram)
- Vocabulary builder
- Simple tokenizer
- Embedding similarity search

**Real-world applications:** NLP preprocessing, semantic search

---

### s19: Sequence-to-Sequence Models (3-4 days) ⭐⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Encoder-decoder architecture
- Teacher forcing
- Beam search decoding
- Attention in seq2seq
- Evaluation metrics (BLEU, perplexity)

**Key Implementations:**
- Seq2Seq model (RNN-based and Transformer-based)
- Beam search decoder
- Greedy and sampling decoding
- BLEU score calculator
- Training with teacher forcing

**Real-world applications:** Machine translation, text summarization, chatbots

---

## Part 5: Reinforcement Learning (Weeks 10-11)

### s20: Markov Decision Processes (3-4 days) ⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- MDP formulation (states, actions, rewards, transitions)
- Value functions (state-value, action-value)
- Bellman equations
- Value iteration algorithm
- Policy iteration algorithm
- Gridworld environments

**Key Implementations:**
- Simple Gridworld environment
- Value iteration solver
- Policy iteration solver
- MDP visualization
- Optimal policy extraction

**Real-world applications:** Game playing, robotics, resource allocation

---

### s21: Q-Learning & Temporal Difference (3-4 days) ⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Temporal Difference (TD) learning
- Q-Learning algorithm
- Epsilon-greedy exploration
- Experience replay basics
- Deep Q-Networks (DQN) introduction
- Frozen Lake / CartPole environments

**Key Implementations:**
- Tabular Q-Learning
- Simple DQN (neural Q-function)
- Epsilon-greedy policy
- Basic experience replay buffer
- Training loop with environment interaction

**Real-world applications:** Game AI, autonomous navigation

---

### s22: Policy Gradient Methods (4-5 days) ⭐⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Policy-based vs value-based RL
- REINFORCE algorithm
- Policy gradient theorem
- Baseline (value function as baseline)
- Advantage functions
- Continuous action spaces

**Key Implementations:**
- Policy network (stochastic policies)
- REINFORCE with baseline
- Advantage estimation
- Continuous action sampling
- Training on CartPole/LunarLander

**Real-world applications:** Robotics control, continuous control tasks

---

### s23: Actor-Critic & Advanced RL (4-5 days) ⭐⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Actor-Critic architecture
- Advantage Actor-Critic (A2C)
- Proximal Policy Optimization (PPO) intuition
- Generalized Advantage Estimation (GAE)
- Parallel environment training

**Key Implementations:**
- A2C algorithm
- PPO basics (clipped objective)
- GAE calculator
- Multi-environment wrapper
- Complete training pipeline

**Real-world applications:** Complex control, multi-agent systems

---

## Part 6: Generative AI (Weeks 12-15)

### s24: Variational Autoencoders (4-5 days) ⭐⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Autoencoder basics (encoder-decoder)
- Latent space representation
- Variational inference
- Reparameterization trick
- KL divergence loss
- VAE training (ELBO objective)
- Latent space interpolation

**Key Implementations:**
- `Encoder` and `Decoder` networks
- Reparameterization layer
- VAE loss (reconstruction + KL)
- Complete VAE model
- Latent space visualization and sampling

**Prerequisites:** s12 (Neural Networks), s04 (Probability)

**Real-world applications:** Image generation, anomaly detection, data compression

---

### s25: Generative Adversarial Networks (5-6 days) ⭐⭐⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- GAN framework (generator vs discriminator)
- Adversarial training
- Minimax objective
- Training instabilities
- Deep Convolutional GAN (DCGAN)
- Mode collapse and solutions
- Wasserstein GAN intuition

**Key Implementations:**
- `Generator` network
- `Discriminator` network
- Adversarial training loop
- DCGAN architecture
- Loss variations (vanilla, WGAN)
- FID score calculator (optional)

**Real-world applications:** Image synthesis, data augmentation, style transfer

---

### s26: Diffusion Models (5-6 days) ⭐⭐⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Diffusion process (forward/reverse)
- Denoising Diffusion Probabilistic Models (DDPM)
- Noise schedule (linear, cosine)
- U-Net architecture for denoising
- Training objective (noise prediction)
- Sampling process (reverse diffusion)
- Classifier-free guidance basics

**Key Implementations:**
- Forward diffusion (noise addition)
- Simple U-Net architecture
- DDPM training objective
- Reverse sampling process
- Noise schedule implementations
- Conditional generation basics

**Prerequisites:** s13 (CNNs), s04 (Probability), strong understanding of s12

**Real-world applications:** State-of-the-art image generation (Stable Diffusion, DALL-E), video generation

---

### s27: Large Language Models (6-7 days) ⭐⭐⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Transformer-based LLM architecture
- Causal (autoregressive) language modeling
- GPT architecture
- Training on text corpora
- Next-token prediction
- Sampling strategies (greedy, top-k, nucleus/top-p)
- Temperature scaling
- Context windows and KV caching basics

**Key Implementations:**
- GPT-style decoder-only Transformer
- Causal attention mask
- Language modeling loss (cross-entropy)
- Text generation with various sampling
- Simple tokenizer (BPE basics or character-level)
- Training loop on small text dataset

**Prerequisites:** s16 (Transformers), s18 (Embeddings)

**Real-world applications:** ChatGPT, Claude, GPT-4, code generation, chatbots

---

### s28: RLHF & Fine-tuning (5-6 days) ⭐⭐⭐⭐⭐
**Status:** 🔄 To be created

**Topics:**
- Fine-tuning pre-trained models
- Instruction tuning
- Reinforcement Learning from Human Feedback (RLHF)
- Reward modeling
- Proximal Policy Optimization for LLMs
- Direct Preference Optimization (DPO) intuition
- LoRA (Low-Rank Adaptation) basics
- Safety and alignment

**Key Implementations:**
- Fine-tuning loop (supervised)
- Reward model training
- RLHF pipeline (simplified PPO for language)
- Preference dataset handling
- LoRA layer implementation
- Instruction following evaluation

**Prerequisites:** s22 (Policy Gradients), s23 (Actor-Critic), s27 (LLMs)

**Real-world applications:** ChatGPT training process, custom AI assistants, aligned AI systems

---

## Learning Paths

### Fast Track (Focus on Deep Learning & GenAI)
Skip or skim: s08, s09, s10 (traditional ML)
Focus: s01-s07, s11-s28

### Traditional ML Focus
Complete: s01-s10 in depth
Optional: s11-s28

### Researcher Path
Complete all stages with optional extensions
Implement paper variations for each stage

### Engineer Path
Complete s01-s07, s11-s19, s27-s28
Focus on practical implementations

---

## Project Milestones

### Milestone 1 (Week 4): Complete ML Project
Build an end-to-end ML pipeline using s01-s09 implementations

### Milestone 2 (Week 7): Image Classifier
Train a CNN from scratch on CIFAR-10 or similar dataset

### Milestone 3 (Week 9): Sequence Model
Build a machine translation or text generation model

### Milestone 4 (Week 11): RL Agent
Train an agent to play a game using policy gradients

### Milestone 5 (Week 15): Generative Model
Train a small-scale generative model (VAE, GAN, or diffusion)

### Final Project (Week 16+): Custom GenAI Application
Combine multiple concepts to build a unique application

---

## Assessment & Grading

Each stage is graded using:
```bash
python scripts/grade.py s<XX>_<stage_name>
```

**Grading criteria:**
- **70%**: Test pass rate (correctness)
- **15%**: Numerical stability (edge cases)
- **15%**: Code quality (readability, efficiency)

**Target scores:**
- 85%+: Ready for next stage
- 70-84%: Review and improve
- <70%: Re-attempt with more practice

---

## Resources

### Books
- **Deep Learning** (Goodfellow, Bengio, Courville) - Comprehensive DL textbook
- **Pattern Recognition and Machine Learning** (Bishop) - Mathematical foundations
- **Dive into Deep Learning** (Zhang et al.) - Free, code-focused
- **Reinforcement Learning** (Sutton & Barto) - RL bible

### Online Courses
- **Stanford CS231n** - CNNs for Visual Recognition
- **Stanford CS224n** - NLP with Deep Learning
- **Berkeley CS285** - Deep Reinforcement Learning
- **Fast.ai** - Practical deep learning

### Papers (Read as you progress)
- Attention Is All You Need (Transformers) - After s15
- BERT, GPT-2, GPT-3 papers - After s27
- DDPM, Stable Diffusion papers - After s26
- PPO, DQN papers - After s22-s23

---

## Next Steps

After completing this curriculum, you'll be ready to:
1. Read and implement recent ML/AI papers
2. Contribute to open-source ML libraries
3. Build production ML systems
4. Pursue advanced research or engineering roles
5. Design custom architectures for specific problems

## Getting Started

```bash
# Start with stage 1
cd stages/s01_number_systems

# Read the materials
cat README.md
cat spec.md

# Implement the TODOs
vim starter/*.py

# Test your implementation
pytest tests/ -v
python scripts/grade.py s01_number_systems

# Repeat for each stage!
```

---

**Good luck on your journey from matrices to generative AI! 🚀**
