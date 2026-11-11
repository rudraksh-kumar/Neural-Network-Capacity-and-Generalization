"""Print compact summary tables of all generated JSON logs in results/experiment_logs/."""

from __future__ import annotations

import json
from pathlib import Path

logs_dir = Path(__file__).resolve().parent.parent / "results" / "experiment_logs"


def show(name: str):
    p = logs_dir / name
    return json.load(open(p, "r", encoding="utf-8")) if p.exists() else None


e1 = show("E1_gradient_check.json")
if e1:
    print(f"\n=== E1 Gradient Verification (all_passed={e1['all_passed']}) ===")
    for chk in e1["checks"]:
        max_fd = max(v["rel_err_fd"] for v in chk["parameters"].values())
        max_pt = max(v["rel_err_pt"] for v in chk["parameters"].values())
        print(f"  act={chk['activation']:<4s} lam={chk['l2_lambda']:<4} | max_rel_FD={max_fd:.2e} | max_rel_PT={max_pt:.2e}")

e2 = show("E2_baseline.json")
if e2:
    print("\n=== E2 Baseline Comparison ===")
    for key, agg in e2["aggregates"].items():
        print(f"  {key:<22s} | train={agg['train']['mse_mean']:.4f} | val={agg['val']['mse_mean']:.4f} | test={agg['test']['mse_mean']:.4f} | R2={agg['test']['r2_mean']:.4f}")

e3_syn = show("E3_capacity_synthetic.json")
if e3_syn:
    print("\n=== E3 Capacity (Synthetic) ===")
    print(f"Best H={e3_syn['best_H']} | Test MSE={e3_syn['selected_test_aggregate']['mse_mean']:.5f} | Test R2={e3_syn['selected_test_aggregate']['r2_mean']:.4f}")
    for r in e3_syn["width_results"]:
        print(f"  H={r['hidden_width']:3d} P={r['n_params']:4d} | train={r['train_aggregate']['mse_mean']:.4f} | val={r['val_aggregate']['mse_mean']:.4f}±{r['val_aggregate']['mse_std']:.4f} | gap={r['gap_aggregate']['mse_gap_mean']:.4f}")

e3_con = show("E3_capacity_concrete.json")
if e3_con:
    print("\n=== E3 Capacity (Concrete MPa) ===")
    print(f"Best H={e3_con['best_H']} | Test MSE={e3_con['selected_test_aggregate']['mse_mean']:.4f} (RMSE={e3_con['selected_test_aggregate']['rmse_mean']:.4f}) | Test R2={e3_con['selected_test_aggregate']['r2_mean']:.4f}")
    for r in e3_con["width_results"]:
        print(f"  H={r['hidden_width']:3d} P={r['n_params']:4d} | train={r['train_aggregate']['mse_mean']:.4f} | val={r['val_aggregate']['mse_mean']:.4f}±{r['val_aggregate']['mse_std']:.4f} | gap={r['gap_aggregate']['mse_gap_mean']:.4f}")

for ds in ("synthetic", "concrete"):
    e4 = show(f"E4_sample_size_{ds}.json")
    if e4:
        print(f"\n=== E4 Sample Size ({ds}) ===")
        for r in e4["size_results"]:
            print(f"  N={r['n_train']:4d} | train={r['train_aggregate']['mse_mean']:.4f} | val={r['val_aggregate']['mse_mean']:.4f}±{r['val_aggregate']['mse_std']:.4f} | test={r['test_aggregate']['mse_mean']:.4f} | gap={r['gap_aggregate']['mse_gap_mean']:.4f} | lin_val={r['linear_val_aggregate']['mse_mean']:.4f}")

for ds in ("synthetic", "concrete"):
    e5 = show(f"E5_regularization_{ds}.json")
    if e5:
        print(f"\n=== E5 Regularization ({ds}) ===")
        print(f"  Best overall={e5['best_overall_label']} | post-sel test MSE={e5['post_selection_test'][e5['best_overall_label']]['mse_mean']:.4f} (R2={e5['post_selection_test'][e5['best_overall_label']]['r2_mean']:.4f})")
        for r in e5["reg_results"]:
            print(f"  {r['label']:<25s} | train={r['train_aggregate']['mse_mean']:.4f} | val={r['val_aggregate']['mse_mean']:.4f}±{r['val_aggregate']['mse_std']:.4f} | gap={r['gap_aggregate']['mse_gap_mean']:.4f} | ||W||^2={r['weight_norm_aggregate']['weight_l2_norm_sq_mean']:.1f}")

for ds in ("synthetic", "concrete"):
    e6 = show(f"E6_randomized_{ds}.json")
    if e6:
        print(f"\n=== E6 Randomized Labels ({ds}) ===")
        for r in e6["results"]:
            if r["n_train"] in (25, 100, 200):
                print(f"  {r['labels']:<6s} N={r['n_train']:3d} H={r['hidden_width']:3d} | train={r['train_aggregate']['mse_mean']:.4f} | val={r['val_aggregate']['mse_mean']:.4f} | test={r['test_aggregate']['mse_mean']:.4f} | mem_rate={r['memorization_rate']:.2f} | ep_fit={r['mean_epochs_to_fit']}")
        if "spectral_bias_diagnostic" in e6:
            print("  Spectral bias diagnostic (1D H=256):")
            for r in e6["spectral_bias_diagnostic"]:
                print(f"    N={r['n_train']:3d} {r['init_mode']:<22s} | train={r['train_aggregate']['mse_mean']:.4f} | val={r['val_aggregate']['mse_mean']:.4f}")

for ds in ("synthetic", "concrete"):
    e7 = show(f"E7_budget_{ds}.json")
    if e7:
        print(f"\n=== E7 Budgets ({ds}) ===")
        for r in e7["protocol_a"]["fixed_step_results"]:
            print(f"  A-FixedStep H={r['hidden_width']:3d} P={r['n_params']:4d} st={r['steps_run']:4d} FLOPs={r['total_flops']:.2e} val={r['val_aggregate']['mse_mean']:.4f}")
        for r in e7["protocol_a"]["matched_flop_results"]:
            print(f"  A-MatchFLOP H={r['hidden_width']:3d} P={r['n_params']:4d} st={r['steps_run']:4d} FLOPs={r['total_flops']:.2e} val={r['val_aggregate']['mse_mean']:.4f}")
        for r in e7["protocol_b"]["results"]:
            print(f"  B-Converge  H={r['hidden_width']:3d} ep={r['mean_epochs']:6.1f} conv={r['convergence_rate']:.2f} val={r['val_aggregate']['mse_mean']:.4f}")

for ds in ("synthetic", "concrete"):
    e8 = show(f"E8_diagnostics_{ds}.json")
    if e8:
        print(f"\n=== E8 Diagnostics ({ds}) ===")
        for r in e8["lr_sweep"]["results"]:
            print(f"  LR={r['lr']:.0e} | train={r['train_aggregate']['mse_mean']:.4f} | val={r['val_aggregate']['mse_mean']:.4f}")
        for r in e8["optimizer_sweep"]["results"]:
            print(f"  Opt={r['optimizer']:<10s} | train={r['train_aggregate']['mse_mean']:.4f} | val={r['val_aggregate']['mse_mean']:.4f}")
        for r in e8["init_sweep"]["results"]:
            print(f"  Init={r['init_label']:<12s} | train={r['train_aggregate']['mse_mean']:.4f} | val={r['val_aggregate']['mse_mean']:.4f}")
        for r in e8["overfitting_interventions"]["results"]:
            print(f"  Overfit={r['label']:<22s} | train={r['train_aggregate']['mse_mean']:.4f} | val={r['val_aggregate']['mse_mean']:.4f}")
        for r in e8["representation_interventions"]["results"]:
            print(f"  Repr={r['label']:<18s} | train={r['train_aggregate']['mse_mean']:.4f} | val={r['val_aggregate']['mse_mean']:.4f}")

e9 = show("E9_extensions.json")
if e9:
    print("\n=== E9A Early-Stopping Warmup Ablation ===")
    for r in e9["E9A_warmup"]["results"]:
        print(f"  {r['label']:<14s} | stop={r['mean_stopped_epoch']:6.1f} | best={r['mean_best_val_epoch']:6.1f} | train={r['train_aggregate']['mse_mean']:.4f} | val={r['val_aggregate']['mse_mean']:.4f}±{r['val_aggregate']['mse_std']:.4f}")

    print("\n=== E9B Mini-Batch Implicit Regularization (Concrete) ===")
    for r in e9["E9B_minibatch"]["results"]:
        print(f"  opt={r['optimizer']:<8s} H={r['hidden_width']:>3d} B={r['batch_label']:<9s} | train={r['train_aggregate']['mse_mean']:5.2f} | val={r['val_aggregate']['mse_mean']:5.2f}±{r['val_aggregate']['mse_std']:4.2f} | test_R2={r['test_aggregate']['r2_mean']:.4f}")

    print("\n=== E9C 5-Fold Cross-Validation (Concrete) ===")
    for r in e9["E9C_kfold"]["results"]:
        fa = r["across_fold_aggregate"]
        print(f"  {r['model']:<28s} | RMSE={fa['rmse_mean']:.3f}±{fa['rmse_std']:.3f} MPa | R2={fa['r2_mean']:.4f}±{fa['r2_std']:.4f}")

    print("\n=== E9D Extreme Double Descent & 2-HL Depth (Concrete) ===")
    for r in e9["E9D_double_descent"]["one_hidden_layer_results"]:
        print(f"  1-HL H={r['hidden_width']:>4d} (P={r['n_params']:>5d}, P/N={r['p_over_n']:5.2f}) | train={r['train_aggregate']['mse_mean']:5.2f} | val={r['val_aggregate']['mse_mean']:5.2f} | test={r['test_aggregate']['mse_mean']:5.2f} (R2={r['test_aggregate']['r2_mean']:.4f})")
    for r in e9["E9D_double_descent"]["two_hidden_layer_results"]:
        print(f"  {r['arch']:<13s} (P={r['n_params']:>5d}, P/N={r['p_over_n']:5.2f}) | train={r['train_aggregate']['mse_mean']:5.2f} | val={r['val_aggregate']['mse_mean']:5.2f} | test={r['test_aggregate']['mse_mean']:5.2f} (R2={r['test_aggregate']['r2_mean']:.4f})")
