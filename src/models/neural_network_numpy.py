"""Pure-NumPy one-hidden-layer MLP for regression.

Implements forward propagation, backpropagation, and parameter management
with explicit matrix-dimension annotations matching the project derivations.

Architecture:  Input(d) → Dense(H) → φ → Dense(1) → ŷ

Notation (single sample x ∈ ℝ^d, vectorised batch X ∈ ℝ^{B×d}):
    Z₁ = X W₁ᵀ + 1_B b₁ᵀ   ∈ ℝ^{B×H}
    A₁ = φ(Z₁)               ∈ ℝ^{B×H}
    Ŷ  = A₁ W₂ᵀ + 1_B b₂ᵀ   ∈ ℝ^{B×1}
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def _tanh_deriv(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """dφ/dz = 1 - tanh²(z) = 1 - a²  where a = tanh(z)."""
    return 1.0 - a * a


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def _relu_deriv(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """dφ/dz = 1 if z > 0 else 0  (subgradient 0 at z = 0)."""
    return (z > 0).astype(z.dtype)


_ACTIVATIONS = {
    "tanh": (_tanh, _tanh_deriv),
    "relu": (_relu, _relu_deriv),
}


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class NeuralNetworkNumPy:
    """One-hidden-layer MLP implemented entirely in NumPy (float64).

    Parameters
    ----------
    input_dim : int
        Number of input features *d*.
    hidden_width : int
        Number of hidden units *H*.
    activation : str
        ``'tanh'`` or ``'relu'``.
    init_scheme : str
        ``'xavier'`` (Glorot uniform, good for tanh), ``'he'`` (Kaiming normal,
        good for ReLU), ``'normal'`` (zero-mean Gaussian with *init_scale*),
        or ``'auto'`` (picks xavier for tanh, he for relu).
    init_scale : float | None
        Standard deviation when *init_scheme='normal'*.
    seed : int | None
        Random seed for reproducible initialisation.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_width: int = 32,
        activation: str = "tanh",
        init_scheme: str = "auto",
        init_scale: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_width = int(hidden_width)
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{activation}', use {list(_ACTIVATIONS)}")
        self.activation_name = activation
        self._act_fn, self._act_deriv = _ACTIVATIONS[activation]

        # --- Initialise parameters ---
        rng = np.random.default_rng(seed)
        scheme = init_scheme if init_scheme != "auto" else ("xavier" if activation == "tanh" else "he")

        d, H = self.input_dim, self.hidden_width
        if scheme == "xavier":
            limit = np.sqrt(6.0 / (d + H))
            self.W1 = rng.uniform(-limit, limit, size=(H, d)).astype(np.float64)
        elif scheme == "he":
            std = np.sqrt(2.0 / d)
            self.W1 = rng.normal(0.0, std, size=(H, d)).astype(np.float64)
        elif scheme == "normal":
            std = init_scale if init_scale is not None else 0.01
            self.W1 = rng.normal(0.0, std, size=(H, d)).astype(np.float64)
        else:
            raise ValueError(f"Unknown init_scheme '{scheme}'")

        self.b1 = np.zeros((1, H), dtype=np.float64)

        # Output layer: small normal init
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / H), size=(1, H)).astype(np.float64)
        self.b2 = np.zeros((1, 1), dtype=np.float64)

        # Cache for backward pass
        self._cache: dict[str, np.ndarray] = {}

    # ---- public parameter interface ----

    def count_parameters(self) -> int:
        """Total number of trainable scalar parameters P = H(d+2) + 1."""
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def flops_per_step(self, batch_size: int) -> int:
        """Approximate FLOPs per forward+backward pass (≈ 6·B·P)."""
        return 6 * batch_size * self.count_parameters()

    def get_params(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        for name in ("W1", "b1", "W2", "b2"):
            arr = params[name]
            getattr(self, name)[:] = arr

    def get_param_list(self) -> list[tuple[str, np.ndarray]]:
        """Return (name, array) pairs in canonical order."""
        return [("W1", self.W1), ("b1", self.b1), ("W2", self.W2), ("b2", self.b2)]

    def copy_params(self) -> dict[str, np.ndarray]:
        return {name: arr.copy() for name, arr in self.get_param_list()}

    # ---- forward pass ----

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute Ŷ = (φ(X W₁ᵀ + b₁)) W₂ᵀ + b₂ and cache intermediates.

        Parameters
        ----------
        X : ndarray of shape (B, d)

        Returns
        -------
        Y_hat : ndarray of shape (B, 1)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, self.input_dim)

        Z1 = X @ self.W1.T + self.b1          # (B, H)
        A1 = self._act_fn(Z1)                  # (B, H)
        Y_hat = A1 @ self.W2.T + self.b2       # (B, 1)

        self._cache = {"X": X, "Z1": Z1, "A1": A1, "Y_hat": Y_hat}
        return Y_hat

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass without modifying cache (for evaluation)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, self.input_dim)
        Z1 = X @ self.W1.T + self.b1
        A1 = self._act_fn(Z1)
        return A1 @ self.W2.T + self.b2

    # ---- loss ----

    @staticmethod
    def mse_loss(Y_hat: np.ndarray, Y: np.ndarray) -> float:
        """Mean Squared Error = (1/B) Σ (ŷ_i - y_i)²."""
        diff = Y_hat - Y
        return float(np.mean(diff ** 2))

    @staticmethod
    def mse_loss_with_l2(
        Y_hat: np.ndarray,
        Y: np.ndarray,
        W1: np.ndarray,
        W2: np.ndarray,
        l2_lambda: float,
    ) -> float:
        """MSE + (λ/2)(‖W₁‖²_F + ‖W₂‖²_F).  Biases excluded."""
        mse = float(np.mean((Y_hat - Y) ** 2))
        if l2_lambda > 0.0:
            mse += 0.5 * l2_lambda * (float(np.sum(W1 ** 2)) + float(np.sum(W2 ** 2)))
        return mse

    # ---- backward pass ----

    def backward(
        self,
        Y: np.ndarray,
        l2_lambda: float = 0.0,
        regularize_biases: bool = False,
    ) -> dict[str, np.ndarray]:
        """Compute gradients of the (regularised) MSE loss w.r.t. all parameters.

        Derivation (consistent with project report notation):
        ─────────────────────────────────────────────────────
        L = (1/B) Σᵢ (ŷᵢ − yᵢ)²  + (λ/2)(‖W₁‖²_F + ‖W₂‖²_F)

        ∂L/∂Ŷ  = (2/B)(Ŷ − Y)             shape (B, 1)   … "δ_out"
        ∂L/∂W₂ = δ_outᵀ A₁ / 1             → (1, H)      + λ W₂
        ∂L/∂b₂ = Σ δ_out (along batch)      → (1, 1)
        δ₁     = δ_out W₂ ⊙ φ'(Z₁)         → (B, H)      … "δ_hidden"
        ∂L/∂W₁ = δ₁ᵀ X                      → (H, d)      + λ W₁
        ∂L/∂b₁ = Σ δ₁ (along batch)         → (1, H)

        Returns
        -------
        grads : dict  {'W1': ..., 'b1': ..., 'W2': ..., 'b2': ...}
        """
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        X = self._cache["X"]       # (B, d)
        Z1 = self._cache["Z1"]     # (B, H)
        A1 = self._cache["A1"]     # (B, H)
        Y_hat = self._cache["Y_hat"]  # (B, 1)
        B = X.shape[0]

        # ── Output layer gradients ──
        delta_out = (2.0 / B) * (Y_hat - Y)                # (B, 1)
        dW2 = delta_out.T @ A1                               # (1, H)
        db2 = np.sum(delta_out, axis=0, keepdims=True)        # (1, 1)

        # ── Hidden layer gradients ──
        act_prime = self._act_deriv(Z1, A1)                  # (B, H)
        delta_hidden = (delta_out @ self.W2) * act_prime     # (B, H)
        dW1 = delta_hidden.T @ X                              # (H, d)
        db1 = np.sum(delta_hidden, axis=0, keepdims=True)     # (1, H)

        # ── L2 regularisation contribution (weights only by default) ──
        if l2_lambda > 0.0:
            dW1 = dW1 + l2_lambda * self.W1
            dW2 = dW2 + l2_lambda * self.W2
            if regularize_biases:
                db1 = db1 + l2_lambda * self.b1
                db2 = db2 + l2_lambda * self.b2

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ---- convenience ----

    def summary(self) -> str:
        d, H = self.input_dim, self.hidden_width
        P = self.count_parameters()
        return (
            f"NeuralNetworkNumPy(d={d}, H={H}, act={self.activation_name}, P={P})\n"
            f"  W1: ({H}, {d})  b1: (1, {H})  W2: (1, {H})  b2: (1, 1)"
        )

    def __repr__(self) -> str:
        return f"NeuralNetworkNumPy(d={self.input_dim}, H={self.hidden_width}, act={self.activation_name})"

