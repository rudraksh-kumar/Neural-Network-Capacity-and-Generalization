"""L2 weight penalty, Frobenius norm complexity metrics, and bias-exclusion helpers."""

from __future__ import annotations

import numpy as np


def compute_weight_l2_norm_sq(
    W1: np.ndarray,
    W2: np.ndarray,
    b1: np.ndarray | None = None,
    b2: np.ndarray | None = None,
    include_biases: bool = False,
) -> float:
    """Compute squared Frobenius norm ||W1||_F^2 + ||W2||_F^2 (optionally biases)."""
    total = float(np.sum(W1 ** 2) + np.sum(W2 ** 2))
    if include_biases and b1 is not None and b2 is not None:
        total += float(np.sum(b1 ** 2) + np.sum(b2 ** 2))
    return total


def l2_penalty_loss(
    W1: np.ndarray,
    W2: np.ndarray,
    l2_lambda: float,
    b1: np.ndarray | None = None,
    b2: np.ndarray | None = None,
    regularize_biases: bool = False,
) -> float:
    """Compute (lambda / 2) * sum ||W_k||_F^2."""
    if l2_lambda <= 0.0:
        return 0.0
    return 0.5 * float(l2_lambda) * compute_weight_l2_norm_sq(
        W1, W2, b1=b1, b2=b2, include_biases=regularize_biases
    )


def add_l2_gradient(
    grads: dict[str, np.ndarray],
    params: dict[str, np.ndarray],
    l2_lambda: float,
    regularize_biases: bool = False,
) -> dict[str, np.ndarray]:
    """Return new gradient dict with coupled L2 gradient term + lambda * param."""
    if l2_lambda <= 0.0:
        return {k: v.copy() for k, v in grads.items()}
    out: dict[str, np.ndarray] = {}
    for name, g in grads.items():
        if name.startswith("b") and not regularize_biases:
            out[name] = g.copy()
        else:
            out[name] = g + l2_lambda * params[name]
    return out

