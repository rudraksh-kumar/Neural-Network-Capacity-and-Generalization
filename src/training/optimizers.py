"""NumPy optimisers: GD, Momentum SGD, Adam, AdamW.

Each optimiser maintains its own state and applies updates to parameter arrays
in-place. The interface is: init(params) → step(params, grads) → repeat.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    """Base optimizer interface."""

    @abstractmethod
    def step(
        self,
        params: dict[str, np.ndarray],
        grads: dict[str, np.ndarray],
    ) -> None:
        """Update *params* in-place using *grads*."""

    def reset(self) -> None:
        """Reset internal state (for fresh training runs)."""


class GD(Optimizer):
    """Vanilla gradient descent (full-batch or mini-batch)."""

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for name in params:
            params[name] -= self.lr * grads[name]


class MomentumSGD(Optimizer):
    """SGD with classical momentum: v ← μv + g;  θ ← θ − η v."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        self.lr = lr
        self.mu = momentum
        self._v: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for name in params:
            if name not in self._v:
                self._v[name] = np.zeros_like(params[name])
            self._v[name] = self.mu * self._v[name] + grads[name]
            params[name] -= self.lr * self._v[name]

    def reset(self) -> None:
        self._v.clear()


class Adam(Optimizer):
    """Adam optimiser (Kingma & Ba, 2015) with coupled L2 regularisation.

    When l2_lambda > 0, the L2 penalty gradient λ·W has already been added
    to *grads* by the model's backward() call, so the preconditioned adaptive
    step applies to the full regularised gradient — this is **coupled L2**.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._t: int = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self._t += 1
        for name in params:
            g = grads[name]
            if name not in self._m:
                self._m[name] = np.zeros_like(g)
                self._v[name] = np.zeros_like(g)
            self._m[name] = self.beta1 * self._m[name] + (1.0 - self.beta1) * g
            self._v[name] = self.beta2 * self._v[name] + (1.0 - self.beta2) * (g ** 2)
            m_hat = self._m[name] / (1.0 - self.beta1 ** self._t)
            v_hat = self._v[name] / (1.0 - self.beta2 ** self._t)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self) -> None:
        self._m.clear()
        self._v.clear()
        self._t = 0


class AdamW(Optimizer):
    """AdamW optimiser (Loshchilov & Hutter, 2019) with decoupled weight decay.

    Weight decay is applied *directly* to the parameters before the adaptive
    gradient step, bypassing the second-moment scaling. Biases are optionally
    excluded (regularize_biases=False by default).

    Important distinction from Adam + coupled L2:
      - Adam + L2:  θ ← θ − η · m̂/(√v̂ + ε)   where g includes λθ
      - AdamW:      θ ← (1 − η·wd)θ − η · m̂/(√v̂ + ε)   using only data gradient
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        regularize_biases: bool = False,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.regularize_biases = regularize_biases
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._t: int = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self._t += 1
        for name in params:
            g = grads[name]
            if name not in self._m:
                self._m[name] = np.zeros_like(g)
                self._v[name] = np.zeros_like(g)

            # Decoupled weight decay (skip biases unless flag set)
            is_bias = name.startswith("b")
            if self.weight_decay > 0.0 and (not is_bias or self.regularize_biases):
                params[name] *= (1.0 - self.lr * self.weight_decay)

            self._m[name] = self.beta1 * self._m[name] + (1.0 - self.beta1) * g
            self._v[name] = self.beta2 * self._v[name] + (1.0 - self.beta2) * (g ** 2)
            m_hat = self._m[name] / (1.0 - self.beta1 ** self._t)
            v_hat = self._v[name] / (1.0 - self.beta2 ** self._t)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self) -> None:
        self._m.clear()
        self._v.clear()
        self._t = 0


def build_optimizer(name: str, **kwargs) -> Optimizer:
    """Factory function: create an optimizer by name."""
    registry = {
        "gd": GD,
        "momentum": MomentumSGD,
        "adam": Adam,
        "adamw": AdamW,
    }
    cls = registry.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown optimizer '{name}', choose from {list(registry)}")
    # Filter kwargs to only those accepted by the constructor
    import inspect
    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**valid)

