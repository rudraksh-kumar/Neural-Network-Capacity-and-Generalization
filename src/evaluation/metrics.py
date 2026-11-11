"""Regression evaluation metrics, physical-unit evaluation, and multi-seed aggregation."""

from __future__ import annotations

from typing import Any, Iterable
import numpy as np


def to_serializable(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to native JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MSE, RMSE, MAE, and R^2 score for 1D or (N, 1) arrays."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")

    residual = yp - yt
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def evaluate_model_physical(
    model: Any,
    X: np.ndarray,
    y_model_space: np.ndarray,
    data_split: Any,
) -> dict[str, float]:
    """Evaluate model in original physical units (MPa for Concrete, units for Synthetic).

    For datasets with y_scaler (Concrete), predictions and targets are both
    inverse-transformed to MPa so metrics are directly comparable across
    training subsamples and interpretable in physical units.
    """
    y_pred_model = model.predict(X)
    y_true_phys = data_split.inverse_transform_y(y_model_space)
    y_pred_phys = data_split.inverse_transform_y(y_pred_model)
    metrics = compute_regression_metrics(y_true_phys, y_pred_phys)
    metrics["scaled_mse"] = float(np.mean((y_pred_model - y_model_space) ** 2))
    return metrics


def compute_generalization_gap(
    train_metrics: dict[str, float],
    eval_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute generalization gaps (validation/test minus training error)."""
    gap = {
        "mse_gap": float(eval_metrics["mse"] - train_metrics["mse"]),
        "rmse_gap": float(eval_metrics["rmse"] - train_metrics["rmse"]),
        "mae_gap": float(eval_metrics["mae"] - train_metrics["mae"]),
        "r2_gap": float(train_metrics["r2"] - eval_metrics["r2"]),
    }
    if "scaled_mse" in eval_metrics and "scaled_mse" in train_metrics:
        gap["scaled_mse_gap"] = float(eval_metrics["scaled_mse"] - train_metrics["scaled_mse"])
    return gap


def aggregate_seed_metrics(records: Iterable[dict[str, float]]) -> dict[str, float]:
    """Aggregate scalar metric dictionaries across random seeds into mean and std."""
    rows = list(records)
    if not rows:
        return {}
    keys = rows[0].keys()
    summary: dict[str, float] = {"n_runs": float(len(rows))}
    for key in keys:
        vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return summary
