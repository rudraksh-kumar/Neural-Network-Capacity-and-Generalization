"""PyTorch one-hidden-layer MLP for cross-validation of NumPy results."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class NeuralNetworkTorch(nn.Module):
    """PyTorch mirror of the NumPy MLP for independent gradient verification.

    Architecture: Input(d) → Linear(H) → φ → Linear(1)
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_width: int = 32,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_width = hidden_width
        self.activation_name = activation

        self.fc1 = nn.Linear(input_dim, hidden_width, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden_width, 1, dtype=torch.float64)

        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation '{activation}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.fc1(x)
        a1 = self.act(z1)
        return self.fc2(a1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xt = torch.as_tensor(X, dtype=torch.float64)
            if xt.ndim == 1:
                xt = xt.unsqueeze(1)
            return self(xt).numpy()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def copy_params_from_numpy(self, np_model) -> None:
        """Copy parameters from a NeuralNetworkNumPy instance (for gradient comparison)."""
        with torch.no_grad():
            # np_model.W1 is (H, d), PyTorch fc1.weight is (H, d) — same layout
            self.fc1.weight.copy_(torch.from_numpy(np_model.W1))
            self.fc1.bias.copy_(torch.from_numpy(np_model.b1.reshape(-1)))
            self.fc2.weight.copy_(torch.from_numpy(np_model.W2))
            self.fc2.bias.copy_(torch.from_numpy(np_model.b2.reshape(-1)))

    def get_grads_as_dict(self) -> dict[str, np.ndarray]:
        """Extract gradients in the same layout as NeuralNetworkNumPy."""
        return {
            "W1": self.fc1.weight.grad.numpy().copy(),
            "b1": self.fc1.bias.grad.numpy().reshape(1, -1).copy(),
            "W2": self.fc2.weight.grad.numpy().copy(),
            "b2": self.fc2.bias.grad.numpy().reshape(1, 1).copy(),
        }

    def init_matching_numpy(self, seed: int) -> None:
        """Initialize weights in float64 using exact Xavier (tanh) / He (relu) convention."""
        rng = np.random.default_rng(seed)
        d, H = self.input_dim, self.hidden_width
        with torch.no_grad():
            if self.activation_name == "tanh":
                limit = np.sqrt(6.0 / (d + H))
                w1 = rng.uniform(-limit, limit, size=(H, d))
            else:
                w1 = rng.normal(0.0, np.sqrt(2.0 / d), size=(H, d))
            self.fc1.weight.copy_(torch.from_numpy(w1))
            self.fc1.bias.zero_()
            w2 = rng.normal(0.0, np.sqrt(2.0 / H), size=(1, H))
            self.fc2.weight.copy_(torch.from_numpy(w2))
            self.fc2.bias.zero_()


class TwoHiddenLayerTorch(nn.Module):
    """Two-hidden-layer PyTorch MLP for depth-vs-width double-descent comparison."""

    def __init__(
        self,
        input_dim: int,
        h1: int,
        h2: int,
        activation: str = "tanh",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1, dtype=torch.float64)
        self.fc2 = nn.Linear(h1, h2, dtype=torch.float64)
        self.fc3 = nn.Linear(h2, 1, dtype=torch.float64)
        self.act = nn.Tanh() if activation == "tanh" else nn.ReLU()

        rng = np.random.default_rng(seed)
        with torch.no_grad():
            for layer, fin, fout in [(self.fc1, input_dim, h1), (self.fc2, h1, h2)]:
                if activation == "tanh":
                    lim = np.sqrt(6.0 / (fin + fout))
                    layer.weight.copy_(torch.from_numpy(rng.uniform(-lim, lim, size=(fout, fin))))
                else:
                    layer.weight.copy_(torch.from_numpy(rng.normal(0.0, np.sqrt(2.0 / fin), size=(fout, fin))))
                layer.bias.zero_()
            self.fc3.weight.copy_(torch.from_numpy(rng.normal(0.0, np.sqrt(2.0 / h2), size=(1, h2))))
            self.fc3.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xt = torch.as_tensor(X, dtype=torch.float64)
            return self(xt).numpy()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_torch_mlp(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    lr: float = 0.01,
    max_epochs: int = 3000,
    l2_lambda: float = 0.0,
) -> None:
    """Fast full-batch float64 Adam training helper for high-width PyTorch sweeps."""
    Xt = torch.as_tensor(X_train, dtype=torch.float64)
    Yt = torch.as_tensor(y_train, dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="mean")
    model.train()
    for _ in range(max_epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), Yt)
        if l2_lambda > 0.0:
            l2 = sum(p.pow(2).sum() for n, p in model.named_parameters() if "weight" in n)
            loss = loss + 0.5 * l2_lambda * l2
        loss.backward()
        opt.step()


