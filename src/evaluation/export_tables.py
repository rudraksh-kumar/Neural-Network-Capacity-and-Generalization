"""Export aggregated CSV and Markdown tables directly from saved JSON experiment logs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from config.experiment_config import ProjectPaths
from src.evaluation.diagnostics import generate_diagnostic_summary


def _load(logs_dir: Path, fname: str) -> dict | None:
    p = logs_dir / fname
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def export_all_tables(
    logs_dir: str | Path | None = None,
    tables_dir: str | Path | None = None,
) -> list[Path]:
    paths = ProjectPaths()
    paths.ensure_dirs()
    ldir = Path(logs_dir) if logs_dir else paths.results_logs
    tdir = Path(tables_dir) if tables_dir else paths.results_tables
    tdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # E1 Gradient Check table
    e1 = _load(ldir, "E1_gradient_check.json")
    if e1:
        rows = []
        for chk in e1["checks"]:
            for pname, pm in chk["parameters"].items():
                rows.append({
                    "activation": chk["activation"],
                    "l2_lambda": chk["l2_lambda"],
                    "param": pname,
                    "shape": str(pm["shape"]),
                    "max_abs_err_fd": pm["max_abs_err_fd"],
                    "rel_err_fd": pm["rel_err_fd"],
                    "rel_err_pt": pm["rel_err_pt"],
                    "pass_fd": pm["pass_fd"],
                    "pass_pt": pm["pass_pt"],
                })
        p = tdir / "table_e1_gradient_verification.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        written.append(p)

    # E2 Baseline table
    e2 = _load(ldir, "E2_baseline.json")
    if e2:
        rows = []
        for key, agg in e2["aggregates"].items():
            rows.append({
                "config": key,
                "train_mse": agg["train"]["mse_mean"],
                "train_rmse": agg["train"]["rmse_mean"],
                "val_mse": agg["val"]["mse_mean"],
                "val_rmse": agg["val"]["rmse_mean"],
                "test_mse": agg["test"]["mse_mean"],
                "test_rmse": agg["test"]["rmse_mean"],
                "test_mae": agg["test"]["mae_mean"],
                "test_r2": agg["test"]["r2_mean"],
                "test_r2_std": agg["test"]["r2_std"],
            })
        p = tdir / "table_e2_baseline.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        written.append(p)

    # E3 Capacity table
    for dname in ("synthetic", "concrete"):
        e3 = _load(ldir, f"E3_capacity_{dname}.json")
        if e3:
            rows = []
            for r in e3["width_results"]:
                rows.append({
                    "hidden_width": r["hidden_width"],
                    "n_params": r["n_params"],
                    "train_mse_mean": r["train_aggregate"]["mse_mean"],
                    "train_mse_std": r["train_aggregate"]["mse_std"],
                    "val_mse_mean": r["val_aggregate"]["mse_mean"],
                    "val_mse_std": r["val_aggregate"]["mse_std"],
                    "val_rmse_mean": r["val_aggregate"]["rmse_mean"],
                    "val_r2_mean": r["val_aggregate"]["r2_mean"],
                    "mse_gap_mean": r["gap_aggregate"]["mse_gap_mean"],
                })
            p = tdir / f"table_e3_capacity_{dname}.csv"
            pd.DataFrame(rows).to_csv(p, index=False)
            written.append(p)

    # E4 Sample Size table
    for dname in ("synthetic", "concrete"):
        e4 = _load(ldir, f"E4_sample_size_{dname}.json")
        if e4:
            rows = []
            for r in e4["size_results"]:
                rows.append({
                    "n_train": r["n_train"],
                    "train_mse_mean": r["train_aggregate"]["mse_mean"],
                    "train_mse_std": r["train_aggregate"]["mse_std"],
                    "val_mse_mean": r["val_aggregate"]["mse_mean"],
                    "val_mse_std": r["val_aggregate"]["mse_std"],
                    "test_mse_mean": r["test_aggregate"]["mse_mean"],
                    "test_r2_mean": r["test_aggregate"]["r2_mean"],
                    "mse_gap_mean": r["gap_aggregate"]["mse_gap_mean"],
                    "linear_val_mse_mean": r["linear_val_aggregate"]["mse_mean"],
                })
            p = tdir / f"table_e4_sample_size_{dname}.csv"
            pd.DataFrame(rows).to_csv(p, index=False)
            written.append(p)

    # E5 Regularization table
    for dname in ("synthetic", "concrete"):
        e5 = _load(ldir, f"E5_regularization_{dname}.json")
        if e5:
            rows = []
            for r in e5["reg_results"]:
                rows.append({
                    "method": r["label"],
                    "group": r["group"],
                    "strength": r["strength"],
                    "train_mse_mean": r["train_aggregate"]["mse_mean"],
                    "val_mse_mean": r["val_aggregate"]["mse_mean"],
                    "val_mse_std": r["val_aggregate"]["mse_std"],
                    "val_r2_mean": r["val_aggregate"]["r2_mean"],
                    "mse_gap_mean": r["gap_aggregate"]["mse_gap_mean"],
                    "weight_norm_sq_mean": r["weight_norm_aggregate"]["weight_l2_norm_sq_mean"],
                })
            p = tdir / f"table_e5_regularization_{dname}.csv"
            pd.DataFrame(rows).to_csv(p, index=False)
            written.append(p)

    # E6 Randomized Labels table
    for dname in ("synthetic", "concrete"):
        e6 = _load(ldir, f"E6_randomized_{dname}.json")
        if e6:
            rows = []
            for r in e6["results"]:
                rows.append({
                    "labels": r["labels"],
                    "n_train": r["n_train"],
                    "hidden_width": r["hidden_width"],
                    "train_mse_mean": r["train_aggregate"]["mse_mean"],
                    "val_mse_mean": r["val_aggregate"]["mse_mean"],
                    "test_mse_mean": r["test_aggregate"]["mse_mean"],
                    "memorization_rate": r["memorization_rate"],
                    "mean_epochs_to_fit": r["mean_epochs_to_fit"],
                })
            p = tdir / f"table_e6_randomized_{dname}.csv"
            pd.DataFrame(rows).to_csv(p, index=False)
            written.append(p)

    # E7 Budget table
    for dname in ("synthetic", "concrete"):
        e7 = _load(ldir, f"E7_budget_{dname}.json")
        if e7:
            rows = []
            for r in e7["protocol_a"]["fixed_step_results"]:
                rows.append({"protocol": "A_fixed_steps", "H": r["hidden_width"], "P": r["n_params"], "steps": r["steps_run"], "examples_processed": r["examples_processed"], "flops": r["total_flops"], "wall_s": r["mean_wall_time_sec"], "train_mse": r["train_aggregate"]["mse_mean"], "val_mse": r["val_aggregate"]["mse_mean"]})
            for r in e7["protocol_a"]["matched_flop_results"]:
                rows.append({"protocol": "A_matched_flops", "H": r["hidden_width"], "P": r["n_params"], "steps": r["steps_run"], "examples_processed": r["examples_processed"], "flops": r["total_flops"], "wall_s": r["mean_wall_time_sec"], "train_mse": r["train_aggregate"]["mse_mean"], "val_mse": r["val_aggregate"]["mse_mean"]})
            for r in e7["protocol_b"]["results"]:
                rows.append({"protocol": "B_convergence", "H": r["hidden_width"], "P": r["n_params"], "epochs": r["mean_epochs"], "steps": r["mean_steps"], "flops": r["mean_flops"], "wall_s": r["mean_wall_time_sec"], "train_mse": r["train_aggregate"]["mse_mean"], "val_mse": r["val_aggregate"]["mse_mean"], "conv_rate": r["convergence_rate"]})
            p = tdir / f"table_e7_budget_{dname}.csv"
            pd.DataFrame(rows).to_csv(p, index=False)
            written.append(p)

    # E8 Diagnostics Markdown Table
    diag_summary = generate_diagnostic_summary(ldir)
    if diag_summary["rows"]:
        p = tdir / "table_e8_diagnostics.md"
        p.write_text(diag_summary["markdown_table"], encoding="utf-8")
        written.append(p)

    # E9 Extension CSV tables
    e9 = _load(ldir, "E9_extensions.json")
    if e9:
        e9a = e9.get("E9A_warmup", {}).get("results", [])
        if e9a:
            p = tdir / "table_e9a_warmup.csv"
            pd.DataFrame([
                {
                    "label": r["label"], "warmup_epochs": r["warmup_epochs"],
                    "mean_stopped_epoch": r["mean_stopped_epoch"], "mean_best_val_epoch": r["mean_best_val_epoch"],
                    "train_mse": r["train_aggregate"]["mse_mean"],
                    "val_mse": r["val_aggregate"]["mse_mean"], "val_mse_std": r["val_aggregate"]["mse_std"],
                    "test_mse": r["test_aggregate"]["mse_mean"],
                } for r in e9a
            ]).to_csv(p, index=False)
            written.append(p)

        e9b = e9.get("E9B_minibatch", {}).get("results", [])
        if e9b:
            p = tdir / "table_e9b_minibatch.csv"
            pd.DataFrame([
                {
                    "optimizer": r["optimizer"], "H": r["hidden_width"], "batch_size": r["batch_label"],
                    "epochs": r["epochs"], "train_mse": r["train_aggregate"]["mse_mean"],
                    "val_mse": r["val_aggregate"]["mse_mean"], "val_mse_std": r["val_aggregate"]["mse_std"],
                    "test_r2": r["test_aggregate"]["r2_mean"], "mse_gap": r["gap_aggregate"]["mse_gap_mean"],
                } for r in e9b
            ]).to_csv(p, index=False)
            written.append(p)

        e9c = e9.get("E9C_kfold", {}).get("results", [])
        if e9c:
            p = tdir / "table_e9c_kfold.csv"
            pd.DataFrame([
                {
                    "model": r["model"],
                    "fold_rmse_mean": r["across_fold_aggregate"]["rmse_mean"],
                    "fold_rmse_std": r["across_fold_aggregate"]["rmse_std"],
                    "fold_r2_mean": r["across_fold_aggregate"]["r2_mean"],
                    "fold_r2_std": r["across_fold_aggregate"]["r2_std"],
                } for r in e9c
            ]).to_csv(p, index=False)
            written.append(p)

        e9d = e9.get("E9D_double_descent", {})
        if e9d.get("one_hidden_layer_results"):
            p = tdir / "table_e9d_double_descent.csv"
            rows = [
                {
                    "arch": f"1-HL H={r['hidden_width']}", "n_params": r["n_params"], "p_over_n": r["p_over_n"],
                    "train_mse": r["train_aggregate"]["mse_mean"], "val_mse": r["val_aggregate"]["mse_mean"],
                    "test_mse": r["test_aggregate"]["mse_mean"], "test_r2": r["test_aggregate"]["r2_mean"],
                } for r in e9d["one_hidden_layer_results"]
            ] + [
                {
                    "arch": r["arch"], "n_params": r["n_params"], "p_over_n": r["p_over_n"],
                    "train_mse": r["train_aggregate"]["mse_mean"], "val_mse": r["val_aggregate"]["mse_mean"],
                    "test_mse": r["test_aggregate"]["mse_mean"], "test_r2": r["test_aggregate"]["r2_mean"],
                } for r in e9d.get("two_hidden_layer_results", [])
            ]
            pd.DataFrame(rows).to_csv(p, index=False)
            written.append(p)

    print(f"[Tables] Exported {len(written)} tables to {tdir}")
    return written


if __name__ == "__main__":
    export_all_tables()

