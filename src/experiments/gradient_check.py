"""Finite-difference gradient verification for the NumPy MLP.

Compares:
  1. Analytical gradients (backpropagation).
  2. Numerical gradients (central finite differences).
  3. PyTorch autograd gradients (independent cross-check).

Acceptance criteria:
  - Relative error vs finite diff < 1e-7 for tanh, < 1e-5 for ReLU.
  - Relative error vs PyTorch autograd < 1e-11.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.neural_network_numpy import NeuralNetworkNumPy
from src.models.neural_network_torch import NeuralNetworkTorch


def _numerical_gradient(
    model: NeuralNetworkNumPy,
    X: np.ndarray,
    Y: np.ndarray,
    l2_lambda: float,
    epsilon: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Compute numerical gradients by central finite differences.

    For each scalar parameter θ_k:
        g_num_k = [L(θ + ε e_k) - L(θ - ε e_k)] / (2ε)
    """
    num_grads: dict[str, np.ndarray] = {}

    for name, param in model.get_param_list():
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            original = param[idx]

            # f(θ + ε)
            param[idx] = original + epsilon
            y_plus = model.predict(X)
            loss_plus = model.mse_loss_with_l2(y_plus, Y, model.W1, model.W2, l2_lambda)

            # f(θ - ε)
            param[idx] = original - epsilon
            y_minus = model.predict(X)
            loss_minus = model.mse_loss_with_l2(y_minus, Y, model.W1, model.W2, l2_lambda)

            # Restore
            param[idx] = original

            grad[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
            it.iternext()

        num_grads[name] = grad

    return num_grads


def _pytorch_gradients(
    np_model: NeuralNetworkNumPy,
    X: np.ndarray,
    Y: np.ndarray,
    l2_lambda: float,
) -> dict[str, np.ndarray]:
    """Compute gradients using PyTorch autograd as an independent cross-check."""
    pt_model = NeuralNetworkTorch(
        input_dim=np_model.input_dim,
        hidden_width=np_model.hidden_width,
        activation=np_model.activation_name,
    )
    pt_model.copy_params_from_numpy(np_model)
    pt_model.zero_grad()

    Xt = torch.from_numpy(X).double()
    Yt = torch.from_numpy(Y).double()

    Y_hat = pt_model(Xt)
    loss = nn.MSELoss(reduction="mean")(Y_hat, Yt)

    if l2_lambda > 0.0:
        l2_pen = 0.5 * l2_lambda * (
            pt_model.fc1.weight.pow(2).sum() + pt_model.fc2.weight.pow(2).sum()
        )
        loss = loss + l2_pen

    loss.backward()
    return pt_model.get_grads_as_dict()


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """L2 relative error with safe denominator."""
    diff = np.linalg.norm(a - b)
    denom = max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12)
    return float(diff / denom)


def run_gradient_check(
    activation: str = "tanh",
    input_dim: int = 3,
    hidden_width: int = 4,
    batch_size: int = 8,
    l2_lambda: float = 0.0,
    epsilon: float = 1e-6,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Execute the full gradient verification protocol.

    Returns a summary dict with per-parameter errors and pass/fail status.
    """
    rng = np.random.default_rng(seed)

    # Generate small fixed dataset
    X = rng.standard_normal((batch_size, input_dim)).astype(np.float64)
    Y = rng.standard_normal((batch_size, 1)).astype(np.float64)

    # For ReLU, ensure no pre-activation is near zero
    model = NeuralNetworkNumPy(
        input_dim=input_dim,
        hidden_width=hidden_width,
        activation=activation,
        seed=seed,
    )

    if activation == "relu":
        Z1_check = X @ model.W1.T + model.b1
        min_abs = np.min(np.abs(Z1_check))
        attempts = 0
        while min_abs < 1e-3 and attempts < 100:
            # Perturb inputs slightly
            X += rng.normal(0, 0.01, X.shape)
            Z1_check = X @ model.W1.T + model.b1
            min_abs = np.min(np.abs(Z1_check))
            attempts += 1

    # 1. Analytical gradients (backpropagation)
    Y_hat = model.forward(X)
    ana_grads = model.backward(Y, l2_lambda=l2_lambda)

    # 2. Numerical gradients (central finite differences)
    num_grads = _numerical_gradient(model, X, Y, l2_lambda, epsilon)

    # 3. PyTorch autograd gradients
    pt_grads = _pytorch_gradients(model, X, Y, l2_lambda)

    # Compare
    threshold_fd = 1e-7 if activation == "tanh" else 1e-5
    threshold_pt = 1e-11

    results = {
        "activation": activation,
        "input_dim": input_dim,
        "hidden_width": hidden_width,
        "batch_size": batch_size,
        "l2_lambda": l2_lambda,
        "epsilon": epsilon,
        "parameters": {},
        "overall_pass_fd": True,
        "overall_pass_pt": True,
    }

    if verbose:
        header = (
            f"\n{'='*80}\n"
            f"Gradient Check: activation={activation}, d={input_dim}, H={hidden_width}, "
            f"B={batch_size}, λ={l2_lambda}\n"
            f"{'='*80}"
        )
        print(header)
        print(f"{'Param':<6} {'Shape':<12} {'MaxAbsErr(FD)':<16} {'RelErr(FD)':<16} "
              f"{'RelErr(PT)':<16} {'Pass(FD)':<10} {'Pass(PT)':<10}")
        print("-" * 90)

    for name in ["W1", "b1", "W2", "b2"]:
        a = ana_grads[name]
        n = num_grads[name]
        p = pt_grads[name]

        max_abs_err_fd = float(np.max(np.abs(a - n)))
        rel_err_fd = _relative_error(a, n)
        rel_err_pt = _relative_error(a, p)

        pass_fd = rel_err_fd < threshold_fd
        pass_pt = rel_err_pt < threshold_pt

        results["parameters"][name] = {
            "shape": list(a.shape),
            "max_abs_err_fd": max_abs_err_fd,
            "rel_err_fd": rel_err_fd,
            "rel_err_pt": rel_err_pt,
            "pass_fd": pass_fd,
            "pass_pt": pass_pt,
        }

        if not pass_fd:
            results["overall_pass_fd"] = False
        if not pass_pt:
            results["overall_pass_pt"] = False

        if verbose:
            print(
                f"{name:<6} {str(list(a.shape)):<12} {max_abs_err_fd:<16.2e} "
                f"{rel_err_fd:<16.2e} {rel_err_pt:<16.2e} "
                f"{'✓' if pass_fd else '✗':<10} {'✓' if pass_pt else '✗':<10}"
            )

    if verbose:
        print("-" * 90)
        fd_status = "PASS" if results["overall_pass_fd"] else "FAIL"
        pt_status = "PASS" if results["overall_pass_pt"] else "FAIL"
        print(f"Overall finite-diff: {fd_status}  (threshold {threshold_fd:.0e})")
        print(f"Overall PyTorch:     {pt_status}  (threshold {threshold_pt:.0e})")

    return results


def run_full_gradient_verification() -> list[dict]:
    """Run gradient checks for all required configurations."""
    configs = [
        {"activation": "tanh", "l2_lambda": 0.0},
        {"activation": "tanh", "l2_lambda": 0.01},
        {"activation": "relu", "l2_lambda": 0.0},
        {"activation": "relu", "l2_lambda": 0.01},
    ]
    all_results = []
    for cfg in configs:
        result = run_gradient_check(**cfg, verbose=True)
        all_results.append(result)
    return all_results


def run_experiment() -> dict:
    """Run E1 gradient verification and save JSON log."""
    import json
    from config.experiment_config import ProjectPaths
    from src.evaluation.metrics import to_serializable

    paths = ProjectPaths()
    paths.ensure_dirs()
    checks = run_full_gradient_verification()
    all_pass = all(r["overall_pass_fd"] and r["overall_pass_pt"] for r in checks)
    payload = {
        "experiment": "E1_gradient_check",
        "all_passed": all_pass,
        "checks": checks,
    }
    out_path = paths.results_logs / "E1_gradient_check.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2)
    print(f"[E1] Saved -> {out_path}")
    return payload


if __name__ == "__main__":
    res = run_experiment()
    print(f"\n{'='*80}")
    print(f"FINAL VERDICT: {'ALL CHECKS PASSED' if res['all_passed'] else 'SOME CHECKS FAILED'}")
    print(f"{'='*80}")

