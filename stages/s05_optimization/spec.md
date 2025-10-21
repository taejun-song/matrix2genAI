# Stage 05: Optimization Fundamentals

## Overview

Implement optimization algorithms used throughout machine learning.

## Tasks

### Part 1: Gradient Descent (`starter/gradient_descent.py`)

1. **`gradient_descent(f, grad_f, x0, lr, num_iters)`** - Basic gradient descent
2. **`gradient_descent_with_momentum(f, grad_f, x0, lr, momentum, num_iters)`** - GD with momentum
3. **`gradient_descent_with_line_search(f, grad_f, x0, num_iters)`** - GD with backtracking line search

### Part 2: Advanced Optimizers (`starter/optimizers.py`)

1. **`adagrad(f, grad_f, x0, lr, num_iters)`** - AdaGrad optimizer
2. **`rmsprop(f, grad_f, x0, lr, beta, num_iters)`** - RMSProp optimizer
3. **`adam(f, grad_f, x0, lr, beta1, beta2, num_iters)`** - Adam optimizer

### Part 3: Learning Rate Schedules (`starter/schedules.py`)

1. **`step_decay(lr0, epoch, drop_rate, epochs_drop)`** - Step decay schedule
2. **`exponential_decay(lr0, epoch, decay_rate)`** - Exponential decay
3. **`cosine_annealing(lr0, epoch, T_max)`** - Cosine annealing

## Learning Goals

- Understand optimization algorithms used in deep learning
- Learn adaptive learning rate methods
- Implement learning rate schedules
