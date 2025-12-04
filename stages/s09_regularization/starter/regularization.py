from __future__ import annotations

import numpy as np


def ridge_penalty(weights: np.ndarray, alpha: float) -> float:
    """
    Compute L2 (Ridge) regularization penalty: (α/2) × ||w||₂²

    Args:
        weights: Weight vector, shape (n_features,)
        alpha: Regularization strength (α ≥ 0)

    Returns:
        penalty: L2 penalty value (scalar)

    Example:
        >>> weights = np.array([1.0, 2.0, 3.0])
        >>> ridge_penalty(weights, alpha=0.1)
        0.7
    """
    # TODO: return (alpha / 2) * np.sum(weights ** 2)
    raise NotImplementedError


def ridge_gradient(weights: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute gradient of L2 penalty: ∂P/∂w = α × w

    Args:
        weights: Weight vector, shape (n_features,)
        alpha: Regularization strength

    Returns:
        gradient: Gradient of L2 penalty, shape (n_features,)

    Example:
        >>> weights = np.array([1.0, 2.0, 3.0])
        >>> ridge_gradient(weights, alpha=0.1)
        array([0.1, 0.2, 0.3])
    """
    # TODO: return alpha * weights
    raise NotImplementedError


def lasso_penalty(weights: np.ndarray, alpha: float) -> float:
    """
    Compute L1 (Lasso) regularization penalty: α × ||w||₁

    Args:
        weights: Weight vector, shape (n_features,)
        alpha: Regularization strength (α ≥ 0)

    Returns:
        penalty: L1 penalty value (scalar)

    Example:
        >>> weights = np.array([1.0, -2.0, 3.0])
        >>> lasso_penalty(weights, alpha=0.1)
        0.6
    """
    # TODO: return alpha * np.sum(np.abs(weights))
    raise NotImplementedError


def lasso_subgradient(weights: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute subgradient of L1 penalty: ∂P/∂w = α × sign(w)

    Note: At w=0, L1 is not differentiable. We use sign(0)=0.

    Args:
        weights: Weight vector, shape (n_features,)
        alpha: Regularization strength

    Returns:
        subgradient: Subgradient of L1 penalty, shape (n_features,)

    Example:
        >>> weights = np.array([1.0, -2.0, 0.0])
        >>> lasso_subgradient(weights, alpha=0.1)
        array([0.1, -0.1, 0.0])
    """
    # TODO: return alpha * np.sign(weights)
    raise NotImplementedError


def elastic_net_penalty(
    weights: np.ndarray, alpha: float, l1_ratio: float
) -> float:
    """
    Compute ElasticNet penalty: α × [ρ × ||w||₁ + (1-ρ)/2 × ||w||₂²]

    Args:
        weights: Weight vector, shape (n_features,)
        alpha: Overall regularization strength
        l1_ratio: Mix ratio (ρ). 1.0 = pure L1, 0.0 = pure L2

    Returns:
        penalty: ElasticNet penalty value (scalar)

    Example:
        >>> weights = np.array([1.0, 2.0])
        >>> elastic_net_penalty(weights, alpha=1.0, l1_ratio=0.5)
        2.75
    """
    # TODO:
    # l1_penalty = np.sum(np.abs(weights))
    # l2_penalty = np.sum(weights ** 2)
    # return alpha * (l1_ratio * l1_penalty + (1 - l1_ratio) / 2 * l2_penalty)
    raise NotImplementedError


def elastic_net_gradient(
    weights: np.ndarray, alpha: float, l1_ratio: float
) -> np.ndarray:
    """
    Compute gradient of ElasticNet penalty: α × [ρ × sign(w) + (1-ρ) × w]

    Args:
        weights: Weight vector, shape (n_features,)
        alpha: Overall regularization strength
        l1_ratio: Mix ratio (ρ)

    Returns:
        gradient: Gradient of ElasticNet penalty, shape (n_features,)

    Example:
        >>> weights = np.array([1.0, -2.0])
        >>> elastic_net_gradient(weights, alpha=1.0, l1_ratio=0.5)
        array([1.0, -1.5])
    """
    # TODO:
    # l1_grad = np.sign(weights)
    # l2_grad = weights
    # return alpha * (l1_ratio * l1_grad + (1 - l1_ratio) * l2_grad)
    raise NotImplementedError
