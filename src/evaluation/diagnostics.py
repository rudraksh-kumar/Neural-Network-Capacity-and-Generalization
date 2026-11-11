"""Diagnostic evaluation helpers for separating failure modes via controlled interventions.

Distinguishes three root causes of poor model performance:
  1. Optimization failure (model has capacity, fails to fit training set).
  2. Overfitting (training error low, validation error high/diverging).
  3. Inadequate representation (both training and validation errors remain high
     even after well-tuned optimization).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def diagnose_optimization_failure(e8_results: dict[str, Any]) -> dict[str, Any]:
    """Diagnose optimization failure from LR, init-scale, and optimizer interventions."""
    lr_entries = e8_results.get("lr_sweep", {}).get("results", [])
    opt_entries = e8_results.get("optimizer_sweep", {}).get("results", [])
    init_entries = e8_results.get("init_sweep", {}).get("results", [])

    if not lr_entries or not opt_entries:
        return {"diagnosed": False, "reason": "Missing LR/optimizer diagnostic runs"}

    best_lr = min(lr_entries, key=lambda r: r["val_aggregate"].get("mse_mean", float("inf")))
    worst_lr = max(lr_entries, key=lambda r: r["train_aggregate"].get("mse_mean", 0.0))

    best_opt = min(opt_entries, key=lambda r: r["val_aggregate"].get("mse_mean", float("inf")))
    worst_opt = max(opt_entries, key=lambda r: r["train_aggregate"].get("mse_mean", 0.0))

    init_note = ""
    if init_entries:
        best_init = min(init_entries, key=lambda r: r["val_aggregate"].get("mse_mean", float("inf")))
        worst_init = max(init_entries, key=lambda r: r["val_aggregate"].get("mse_mean", 0.0))
        if best_init["init_label"] != worst_init["init_label"]:
            init_note = (
                f"Init scale '{worst_init['init_label']}' val MSE={worst_init['val_aggregate']['mse_mean']:.4f} "
                f"vs '{best_init['init_label']}' val MSE={best_init['val_aggregate']['mse_mean']:.4f}."
            )

    return {
        "problem": "Optimization Failure",
        "evidence": (
            f"lr={worst_lr['lr']:.0e} stalls at high train MSE ({worst_lr['train_aggregate']['mse_mean']:.4f}); "
            f"vanilla opt '{worst_opt['optimizer']}' train MSE={worst_opt['train_aggregate']['mse_mean']:.4f}. {init_note}"
        ).strip(),
        "intervention": f"Tune LR (to {best_lr['lr']:.0e}), switch optimizer to {best_opt['optimizer']}, use Xavier init",
        "result": (
            f"Train MSE drops to {best_lr['train_aggregate']['mse_mean']:.4f}; "
            f"Val MSE drops to {best_lr['val_aggregate']['mse_mean']:.4f}"
        ),
        "diagnosis": "Optimization Failure (capacity existed, optimizer/LR/init prevented convergence)",
    }


def diagnose_overfitting(e8_results: dict[str, Any]) -> dict[str, Any]:
    """Diagnose overfitting from high-capacity unregularized vs regularized/more-data interventions."""
    ov_entries = e8_results.get("overfitting_interventions", {}).get("results", [])
    if not ov_entries:
        return {"diagnosed": False, "reason": "Missing overfitting intervention runs"}

    baseline = next((r for r in ov_entries if r["label"] == "unreg_H128_smallN"), ov_entries[0])
    interventions = [r for r in ov_entries if r["label"] != baseline["label"]]
    best_int = min(interventions, key=lambda r: r["val_aggregate"].get("mse_mean", float("inf")))

    b_tr = baseline["train_aggregate"]["mse_mean"]
    b_val = baseline["val_aggregate"]["mse_mean"]
    i_tr = best_int["train_aggregate"]["mse_mean"]
    i_val = best_int["val_aggregate"]["mse_mean"]

    return {
        "problem": "Overfitting",
        "evidence": f"H=128 unregularized reaches low train MSE ({b_tr:.4f}) while val MSE degrades ({b_val:.4f}); gap={b_val - b_tr:.4f}",
        "intervention": f"Apply {best_int['label']} (weight decay / early stopping / larger N)",
        "result": f"Train MSE={i_tr:.4f}, Val MSE improves from {b_val:.4f} to {i_val:.4f} (gap={i_val - i_tr:.4f})",
        "diagnosis": "Overfitting (excess variance cured by regularization / early stopping / data)",
    }


def diagnose_inadequate_representation(e8_results: dict[str, Any]) -> dict[str, Any]:
    """Diagnose inadequate representation from linear / narrow-width vs wider / nonlinear interventions."""
    rep_entries = e8_results.get("representation_interventions", {}).get("results", [])
    if not rep_entries:
        return {"diagnosed": False, "reason": "Missing representation intervention runs"}

    linear_entry = next((r for r in rep_entries if "linear" in r["label"].lower()), rep_entries[0])
    best_rep = min(rep_entries, key=lambda r: r["val_aggregate"].get("mse_mean", float("inf")))

    l_tr = linear_entry["train_aggregate"]["mse_mean"]
    l_val = linear_entry["val_aggregate"]["mse_mean"]
    b_tr = best_rep["train_aggregate"]["mse_mean"]
    b_val = best_rep["val_aggregate"]["mse_mean"]

    return {
        "problem": "Inadequate Representation",
        "evidence": f"'{linear_entry['label']}' has high train MSE ({l_tr:.4f}) and high val MSE ({l_val:.4f}) with near-zero gap",
        "intervention": f"Increase model expressivity ({best_rep['label']})",
        "result": f"Both train MSE ({b_tr:.4f}) and val MSE ({b_val:.4f}) drop sharply",
        "diagnosis": "Inadequate Representation / Underfitting (structural bias limits hypothesis class)",
    }


def build_diagnostic_table(all_diagnostics: list[dict[str, Any]]) -> str:
    """Format diagnostic rows as a Markdown comparison table."""
    lines = [
        "| Dataset | Observed Problem | Experimental Evidence | Controlled Intervention | Observed Result | Final Diagnosis |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for row in all_diagnostics:
        if "problem" not in row:
            continue
        lines.append(
            f"| {row.get('dataset', '-')} | {row['problem']} | {row['evidence']} | "
            f"{row['intervention']} | {row['result']} | {row['diagnosis']} |"
        )
    return "\n".join(lines)


def generate_diagnostic_summary(experiment_logs_dir: str | Path) -> dict[str, Any]:
    """Load E8 JSON logs and compile structured diagnostics + Markdown table."""
    logs_dir = Path(experiment_logs_dir)
    rows: list[dict[str, Any]] = []

    for dname in ("synthetic", "concrete"):
        p = logs_dir / f"E8_diagnostics_{dname}.json"
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            e8 = json.load(f)

        for fn in (
            diagnose_optimization_failure,
            diagnose_overfitting,
            diagnose_inadequate_representation,
        ):
            diag = fn(e8)
            diag["dataset"] = dname
            rows.append(diag)

    return {
        "rows": rows,
        "markdown_table": build_diagnostic_table(rows),
    }

