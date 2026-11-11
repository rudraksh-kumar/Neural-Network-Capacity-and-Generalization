"""Linear regression baselines: closed-form OLS and optional Ridge."""

from __future__ import annotations

import numpy as np


class LinearBaseline:
    """Ordinary Least Squares (or Ridge) linear regression via normal equations.

    Model:  ŷ = X w + b   (equivalently ŷ = X_aug @ β  with bias column).

    Parameters
    ----------
    l2_lambda : float
        Ridge penalty strength.  0.0 → pure OLS.
    """

    def __init__(self, l2_lambda: float = 0.0) -> None:
        self.l2_lambda = float(l2_lambda)
        self.w: np.ndarray | None = None  # (d, 1)
        self.b: float = 0.0
        self._input_dim: int | None = None

    # ── Fitting ──

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearBaseline":
        """Solve  β = (XᵀX + λI)⁻¹ Xᵀy  using the augmented design matrix."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        B, d = X.shape
        self._input_dim = d

        # Augment with bias column
        X_aug = np.hstack([X, np.ones((B, 1), dtype=np.float64)])  # (B, d+1)

        # Normal equation with optional Ridge term (no penalty on bias)
        XtX = X_aug.T @ X_aug  # (d+1, d+1)
        if self.l2_lambda > 0.0:
            reg = self.l2_lambda * np.eye(d + 1, dtype=np.float64)
            reg[-1, -1] = 0.0  # don't regularise bias
            XtX = XtX + reg

        Xty = X_aug.T @ y  # (d+1, 1)
        beta = np.linalg.solve(XtX, Xty)  # (d+1, 1)

        self.w = beta[:d]
        self.b = float(beta[d, 0])
        return self

    # ── Prediction ──

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Call .fit() before .predict()")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X @ self.w + self.b).reshape(-1, 1)

    # ── Unified interface ──

    def count_parameters(self) -> int:
        if self._input_dim is None:
            return 0
        return int(self._input_dim + 1)  # d weights + 1 bias

    def flops_per_step(self, batch_size: int) -> int:
        return 2 * batch_size * self.count_parameters()

    def summary(self) -> str:
        d = self._input_dim or "?"
        return f"LinearBaseline(d={d}, λ={self.l2_lambda}, P={self.count_parameters()})"

    def __repr__(self) -> str:
        return self.summary()

