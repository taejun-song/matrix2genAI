# Stage 04: Probability and Statistics

**Duration**: 2-3 days
**Prerequisites**: s02, s03
**Difficulty**: ⭐⭐☆☆☆

**The Big Idea:** Machine learning is about learning from uncertain, noisy data. Probability gives us the mathematical tools to reason about uncertainty and make principled decisions.

## Getting Started

### Setup

```bash
# Navigate to this stage
cd stages/s04_probability

# Run tests (using uv - recommended)
uv run pytest tests/ -v

# Or activate venv first
source .venv/bin/activate  # On Unix/macOS
pytest tests/ -v
```

### Files You'll Edit

- `starter/distributions.py` - Gaussian, Bernoulli, multinomial distributions
- `starter/estimators.py` - Mean, variance, covariance, MLE, MAP
- `starter/sampling.py` - Monte Carlo sampling methods

### Quick Test Commands

```bash
# Test specific module
uv run pytest tests/test_distributions.py -v

# Test specific function
uv run pytest tests/test_estimators.py::TestMLE -v

# Run all tests
uv run pytest tests/ -v
```

---

## What You'll Learn

Probability is fundamental to machine learning. You'll implement:
- Common probability distributions
- Statistical estimators (mean, variance, covariance)
- Maximum Likelihood Estimation (MLE)
- Maximum A Posteriori (MAP) estimation

## Conceptual Understanding

### Why Probability in Machine Learning?

**The fundamental problem:** Real-world data is noisy and uncertain.

```
Example: Spam classification
  Email contains "free": Is it spam?

Deterministic approach (fails):
  if "free" in email: classify as spam
  Problem: "Feel free to contact me" → wrongly classified!

Probabilistic approach (works):
  P(spam | contains "free") = 0.8
  P(not spam | contains "free") = 0.2

  We can reason about uncertainty and make better decisions!
```

**Key insight:** ML models don't give absolute answers. They give probabilities!

### Random Variables: Quantifying Uncertainty

**Definition:** A variable whose value is determined by chance.

```
Example: Coin flip
  X = {1 if heads, 0 if tails}

  Probability mass function (PMF):
    P(X = 1) = 0.5  (heads)
    P(X = 0) = 0.5  (tails)

Example: Measuring temperature
  Y = temperature reading (continuous)

  Probability density function (PDF):
    P(20°C ≤ Y ≤ 21°C) = ∫₂₀²¹ f(y) dy

  Why PDF instead of PMF?
    • Infinitely many possible values
    • P(Y = exactly 20.000...°C) = 0
    • Instead, we ask: probability in a range
```

### The Gaussian Distribution: Nature's Favorite

**Why is it everywhere?**

```
Central Limit Theorem:
  Sum of many random variables → approximately Gaussian

  Example: Height of adults
    Height = genetic factors + nutrition + environment + ...
    Result: Bell curve!

  Example: Measurement errors
    True value + sensor noise₁ + sensor noise₂ + ...
    Result: Gaussian distribution around true value
```

**The formula:**
```
N(x | μ, σ²) = (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))

Parameters:
  μ (mu): Mean - center of the distribution
  σ² (sigma squared): Variance - how spread out it is

Properties:
  • 68% of data within ±1σ of μ
  • 95% of data within ±2σ of μ
  • 99.7% of data within ±3σ of μ
```

**ML applications:**
```
• Linear regression assumes Gaussian noise
• Gaussian Naive Bayes classifier
• Gaussian processes
• Variational autoencoders (VAE) use Gaussian latent space
• Weight initialization in neural networks
```

### Bernoulli and Binomial: Modeling Binary Events

**Bernoulli:** Single coin flip
```
X = 1 with probability p (success)
X = 0 with probability 1-p (failure)

Example: Click prediction
  Will user click on ad?
  X = 1 (click) with probability p = 0.05
  X = 0 (no click) with probability 1-p = 0.95

Application: Logistic regression outputs Bernoulli distribution!
```

**Binomial:** Multiple independent coin flips
```
Y = number of successes in n trials

Example: Email campaign
  Send 1000 emails, each has p=0.05 chance of conversion
  Y ~ Binomial(n=1000, p=0.05)

  Expected conversions: E[Y] = n·p = 1000·0.05 = 50
  Variance: Var[Y] = n·p·(1-p) = 1000·0.05·0.95 = 47.5
```

### Multinomial: Modeling Categorical Outcomes

**Generalization of Bernoulli to multiple categories**
```
Example: Document topic classification
  Topics: {sports, politics, entertainment, tech}
  Probabilities: p = [0.3, 0.25, 0.25, 0.2]

  After reading 100 documents, how many in each category?
  X ~ Multinomial(n=100, p=[0.3, 0.25, 0.25, 0.2])

  Expected: [30, 25, 25, 20]

Application: Softmax output in neural networks is multinomial!
```

### Expectation and Variance: Summarizing Distributions

**Expectation (Mean): E[X]**
```
The "average" or "expected" value

Discrete: E[X] = Σ x · P(X = x)
Continuous: E[X] = ∫ x · f(x) dx

Example: Dice roll
  E[X] = 1·(1/6) + 2·(1/6) + ... + 6·(1/6) = 3.5

Interpretation:
  • Center of mass of the distribution
  • Long-run average if you repeat the experiment many times
  • Where you'd "expect" the value to be

Linearity:
  E[aX + b] = a·E[X] + b
  E[X + Y] = E[X] + E[Y]  (even if X and Y are dependent!)
```

**Variance: Var[X]**
```
How spread out is the distribution?

Var[X] = E[(X - μ)²] = E[X²] - (E[X])²

Example: Two investments
  A: Returns 5% ± 1%  (Var = small, safe)
  B: Returns 5% ± 20% (Var = large, risky)

Standard deviation: σ = √Var[X]  (same units as X)

Properties:
  Var[aX + b] = a² · Var[X]  (shifting by b doesn't change spread!)
  If X and Y are independent: Var[X + Y] = Var[X] + Var[Y]
```

### Covariance and Correlation: Measuring Relationships

**Covariance:**
```
Cov[X, Y] = E[(X - μₓ)(Y - μᵧ)]

Interpretation:
  Cov > 0: X and Y tend to increase together (positive relationship)
  Cov < 0: X increases, Y tends to decrease (negative relationship)
  Cov ≈ 0: No linear relationship

Example: Height and weight
  Tall people tend to weigh more → Cov[height, weight] > 0

Problem with covariance: Scale-dependent!
  Cov[height_cm, weight_kg] ≠ Cov[height_inches, weight_lbs]
```

**Correlation:**
```
Corr[X, Y] = Cov[X, Y] / (σₓ · σᵧ)

Properties:
  • Scale-free: always between -1 and +1
  • Corr = +1: Perfect positive linear relationship
  • Corr = 0: No linear relationship
  • Corr = -1: Perfect negative linear relationship

ML application:
  • Feature selection: Remove highly correlated features
  • PCA: Find directions of maximum correlation
  • Correlation matrix for multivariate Gaussian
```

### Maximum Likelihood Estimation (MLE)

**The Question:** Given data, what parameters best explain it?

**The Principle:** Choose parameters that maximize the probability of observing the data.

```
Example: Coin flips
  Data: HHTHT (3 heads, 2 tails out of 5 flips)

  Question: What's the probability p of heads?

  Likelihood function:
    L(p | data) = P(data | p)
                = p³ · (1-p)²

  Find p that maximizes L(p):
    Take derivative, set to 0:
    dL/dp = 3p² · (1-p)² - 2p³ · (1-p) = 0

    Solution: p = 3/5 = 0.6

  MLE estimate: p̂ = (number of heads) / (total flips)

  Intuition: The observed frequency is our best guess!
```

**Why log-likelihood?**
```
Problem: Likelihood = tiny numbers multiplied together
  L(p) = p₁ · p₂ · ... · pₙ → numerical underflow!

Solution: Maximize log-likelihood instead
  log L(p) = log p₁ + log p₂ + ... + log pₙ

  • Maximizing log L is same as maximizing L (log is monotonic)
  • Addition is more numerically stable than multiplication
  • Derivatives are simpler!

This is why ML uses cross-entropy loss (negative log-likelihood)!
```

**MLE for Gaussian:**
```
Given data x₁, x₂, ..., xₙ from N(μ, σ²)

Log-likelihood:
  log L(μ, σ² | data) = -n/2 · log(2πσ²) - Σ(xᵢ - μ)² / (2σ²)

MLE estimates:
  μ̂ = (1/n) Σ xᵢ  (sample mean)
  σ̂² = (1/n) Σ(xᵢ - μ̂)²  (sample variance)

Beautiful: The sample mean and variance are MLE estimators!
```

### Maximum A Posteriori (MAP): Adding Prior Knowledge

**MLE problem:** What if we have very little data?

```
Example: Coin flip
  Flip once → heads

  MLE estimate: p̂ = 1/1 = 1.0 (100% heads!)
  This seems wrong - we know coins are usually fair-ish!
```

**MAP solution:** Use prior knowledge + data

```
Bayes' rule:
  P(parameters | data) ∝ P(data | parameters) · P(parameters)
                         ↑                      ↑
                      likelihood              prior

MAP estimate:
  parameters_MAP = argmax P(data | parameters) · P(parameters)

Example: Coin with prior
  Prior: Beta(α=10, β=10) - centered at p=0.5
  Data: 1 head, 0 tails

  MAP estimate: p̂ = (α + heads - 1) / (α + β + total - 2)
                   = (10 + 1 - 1) / (10 + 10 + 1 - 2)
                   = 10/19 ≈ 0.53

  Much more reasonable than MLE's p̂=1.0!
```

**MLE vs MAP:**
```
MLE:
  • Only uses data
  • Can overfit with small datasets
  • Equivalent to MAP with uniform prior

MAP:
  • Combines prior knowledge with data
  • More robust with small datasets
  • Regularization in ML is MAP estimation!
    L2 regularization = Gaussian prior
    L1 regularization = Laplace prior
```

### Conditional Probability and Bayes' Rule

**Conditional probability:**
```
P(A | B) = probability of A given that B happened

Example: Medical testing
  P(disease | positive test) = ?

  Not the same as:
  P(positive test | disease) = test sensitivity
```

**Bayes' rule: The most important equation in ML**
```
P(A | B) = P(B | A) · P(A) / P(B)

In ML terms:
  P(hypothesis | data) = P(data | hypothesis) · P(hypothesis) / P(data)
      ↑                      ↑                      ↑               ↑
   posterior             likelihood              prior         evidence

Example: Spam classification
  P(spam | "free") = P("free" | spam) · P(spam) / P("free")

  Given:
    P("free" | spam) = 0.8     (80% of spam has "free")
    P("free" | not spam) = 0.1 (10% of ham has "free")
    P(spam) = 0.3              (30% of emails are spam)

  Calculate:
    P("free") = 0.8·0.3 + 0.1·0.7 = 0.31
    P(spam | "free") = 0.8·0.3 / 0.31 = 0.77

  Conclusion: 77% chance of spam given the word "free"
```

**Applications:**
```
• Naive Bayes classifier
• Bayesian networks
• Particle filters
• Bayesian optimization
• Variational inference
```

## Why This Matters

- **Probabilistic models**: Bayesian networks, HMMs
- **Loss functions**: Cross-entropy is negative log-likelihood!
- **Uncertainty**: Confidence intervals, Bayesian inference
- **Generative models**: VAEs, diffusion models use probability
- **Regularization**: L2/L1 are MAP estimation with priors

## Next Stage

**s05: Optimization Fundamentals**
