"""Modular backpropagation and activation derivative utilities.

Provides standalone functional implementations of the forward and backward
equations documented in the mathematical derivations so they can be inspected,
unit-tested, and reused cleanly.
"""

from __future__ import annotations

import numpy as np


def activation_forward(z: np.ndarray, activation: str) -> np.ndarray:
    """Apply hidden-layer nonlinearity phi(z)."""
    if activation == "tanh":
        return np.tanh(z)
    if activation == "relu":
        return np.maximum(z, 0.0)
    raise ValueError(f"Unsupported activation '{activation}'")


def activation_derivative(z: np.ndarray, a: np.ndarray, activation: str) -> np.ndarray:
    """Compute elementwise derivative phi'(z) using pre-activation z and activation a."""
    if activation == "tanh":
        return 1.0 - a * a
    if activation == "relu":
        return (z > 0.0).astype(z.dtype)
    raise ValueError(f"Unsupported activation '{activation}'")


def compute_backprop_gradients(
    X: np.ndarray,
    Y: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    Z1: np.ndarray,
    A1: np.ndarray,
    Y_hat: np.ndarray,
    activation: str = "tanh",
) -> dict[str, np.ndarray]:
    """Compute unregularized MSE gradients via exact chain rule.

    Notation:
      B = batch size, d = input_dim, H = hidden_width
      L_MSE = (1/B) sum_i (y_hat_i - y_i)^2
    """
    B = X.shape[0]
    delta_out = (2.0 / B) * (Y_hat - Y)          # (B, 1)
    dW2 = delta_out.T @ A1                        # (1, H)
    db2 = np.sum(delta_out, axis=0, keepdims=True)  # (1, 1)

    phi_prime = activation_derivative(Z1, A1, activation)  # (B, H)
    delta_hidden = (delta_out @ W2) * phi_prime   # (B, H)
    dW1 = delta_hidden.T @ X                      # (H, d)
    db1 = np.sum(delta_hidden, axis=0, keepdims=True)  # (1, H)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

