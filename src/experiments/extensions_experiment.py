"""E9: Scientific Extensions — Warmup Early Stopping, Mini-Batch SGD, 5-Fold CV, and Double Descent."""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from config.experiment_config import (
    BATCH_SIZE_GRID,
    EXTENDED_WIDTH_GRID,
    INIT_SEEDS_REAL,
    INIT_SEEDS_SYN,
    ProjectPaths,
    RealDataConfig,
    SyntheticDataConfig,
    TrainConfig,
    WARMUP_GRID,
)
from src.data.real_data import load_concrete_dataset, load_concrete_kfold
from src.data.synthetic_data import build_synthetic_dataset
from src.evaluation.metrics import (
    aggregate_seed_metrics,
    compute_generalization_gap,
    evaluate_model_physical,
    to_serializable,
)
from src.models.linear_baseline import LinearBaseline
from src.models.neural_network_numpy import NeuralNetworkNumPy
from src.models.neural_network_torch import (
    NeuralNetworkTorch,
    TwoHiddenLayerTorch,
    train_torch_mlp,
)
from src.training.trainer import train_model


def run_e9a_warmup_ablation() -> dict:
    """E9A: Early-stopping warmup ablation on small-N synthetic data (N=50, H=128)."""
    print("\n[E9A] Early-stopping warmup ablation (Synthetic N=50, H=128) ...")
    syn_data = build_synthetic_dataset(cfg=SyntheticDataConfig(), n_train=50)
    seeds = INIT_SEEDS_SYN[:5]
    results: list[dict] = []

    configs = [("no_early_stop", False, 0)] + [
        (f"warmup={w}", True, w) for w in WARMUP_GRID
    ]
    for label, use_es, warmup in configs:
        tr_list, val_list, test_list, stops, bests = [], [], [], [], []
        for seed in seeds:
            mlp = NeuralNetworkNumPy(input_dim=1, hidden_width=128, activation="tanh", seed=seed)
            cfg = TrainConfig(
                optimizer="adam",
                learning_rate=0.01,
                max_epochs=2500,
                early_stopping=use_es,
                patience=50,
                min_delta=1e-5,
                warmup_epochs=warmup,
                restore_best=True,
                eval_val_every=1 if use_es else 10,
            )
            hist = train_model(mlp, syn_data.X_train, syn_data.y_train, syn_data.X_val, syn_data.y_val, cfg)
            tr_list.append(evaluate_model_physical(mlp, syn_data.X_train, syn_data.y_train, syn_data))
            val_list.append(evaluate_model_physical(mlp, syn_data.X_val, syn_data.y_val, syn_data))
            test_list.append(evaluate_model_physical(mlp, syn_data.X_test, syn_data.y_test, syn_data))
            stops.append(hist.stopped_epoch if hist.stopped_epoch is not None else hist.epochs_run)
            bests.append(hist.best_val_epoch if hist.best_val_epoch is not None else hist.epochs_run)

        tr_agg = aggregate_seed_metrics(tr_list)
        val_agg = aggregate_seed_metrics(val_list)
        test_agg = aggregate_seed_metrics(test_list)
        results.append({
            "label": label,
            "early_stopping": use_es,
            "warmup_epochs": warmup,
            "mean_stopped_epoch": float(np.mean(stops)),
            "mean_best_val_epoch": float(np.mean(bests)),
            "train_aggregate": tr_agg,
            "val_aggregate": val_agg,
            "test_aggregate": test_agg,
        })
        print(
            f"  {label:<14s} | stop_ep={np.mean(stops):6.1f} | best_ep={np.mean(bests):6.1f} | "
            f"train={tr_agg['mse_mean']:.4f} | val={val_agg['mse_mean']:.4f}±{val_agg['mse_std']:.4f} | test={test_agg['mse_mean']:.4f}"
        )
    return {"experiment": "E9A_early_stop_warmup", "n_train": 50, "hidden_width": 128, "results": results}


def run_e9b_minibatch_sgd() -> dict:
    """E9B: Mini-batch SGD/Momentum vs full-batch Adam implicit regularization on Concrete."""
    print("\n[E9B] Mini-batch SGD/Momentum vs Adam on Concrete (H=64, 128, 256) ...")
    rd = load_concrete_dataset(cfg=RealDataConfig())
    seeds = INIT_SEEDS_REAL[:3]
    widths = [64, 128, 256]
    rows: list[dict] = []

    for opt_name in ("momentum", "adam"):
        for H in widths:
            for bs in BATCH_SIZE_GRID:
                epochs = 800 if bs is not None else 2500
                cfg = TrainConfig(
                    optimizer=opt_name,
                    learning_rate=0.01,
                    momentum=0.9,
                    batch_size=bs,
                    max_epochs=epochs,
                    eval_val_every=25,
                )
                tr_list, val_list, test_list, gap_list = [], [], [], []
                for seed in seeds:
                    mlp = NeuralNetworkNumPy(input_dim=rd.n_features, hidden_width=H, activation="tanh", seed=seed)
                    train_model(mlp, rd.X_train, rd.y_train, rd.X_val, rd.y_val, cfg)
                    tr_m = evaluate_model_physical(mlp, rd.X_train, rd.y_train, rd)
                    val_m = evaluate_model_physical(mlp, rd.X_val, rd.y_val, rd)
                    test_m = evaluate_model_physical(mlp, rd.X_test, rd.y_test, rd)
                    tr_list.append(tr_m)
                    val_list.append(val_m)
                    test_list.append(test_m)
                    gap_list.append(compute_generalization_gap(tr_m, val_m))

                bs_label = str(bs) if bs is not None else "full(618)"
                tr_agg = aggregate_seed_metrics(tr_list)
                val_agg = aggregate_seed_metrics(val_list)
                test_agg = aggregate_seed_metrics(test_list)
                gap_agg = aggregate_seed_metrics(gap_list)
                rows.append({
                    "optimizer": opt_name,
                    "hidden_width": H,
                    "batch_size": bs,
                    "batch_label": bs_label,
                    "epochs": epochs,
                    "train_aggregate": tr_agg,
                    "val_aggregate": val_agg,
                    "test_aggregate": test_agg,
                    "gap_aggregate": gap_agg,
                })
                print(
                    f"  opt={opt_name:<8s} H={H:>3d} B={bs_label:<9s} | "
                    f"train={tr_agg['mse_mean']:5.2f} | val={val_agg['mse_mean']:5.2f}±{val_agg['mse_std']:4.2f} | "
                    f"test_R2={test_agg['r2_mean']:.4f} | gap={gap_agg['mse_gap_mean']:5.2f}"
                )
    return {"experiment": "E9B_minibatch_implicit_reg", "results": rows}


def run_e9c_concrete_kfold() -> dict:
    """E9C: Outer 5-fold cross-validation on UCI Concrete (split + init variance)."""
    print("\n[E9C] Outer 5-fold cross-validation on Concrete ...")
    folds = load_concrete_kfold(n_splits=5, cfg=RealDataConfig())
    seeds = INIT_SEEDS_REAL[:3]
    specs = [
        ("Linear OLS", "linear", 0, None),
        ("MLP H=16 (unreg)", "mlp", 16, TrainConfig(optimizer="adam", learning_rate=0.01, max_epochs=2500, eval_val_every=20)),
        ("MLP H=128 (unreg)", "mlp", 128, TrainConfig(optimizer="adam", learning_rate=0.01, max_epochs=2500, eval_val_every=20)),
        ("MLP H=128 (L2=0.001)", "mlp", 128, TrainConfig(optimizer="adam", learning_rate=0.01, max_epochs=2500, l2_lambda=0.001, eval_val_every=20)),
        ("MLP H=128 (EarlyStop+W100)", "mlp", 128, TrainConfig(optimizer="adam", learning_rate=0.01, max_epochs=2500, early_stopping=True, patience=60, min_delta=1e-5, warmup_epochs=100, restore_best=True)),
    ]

    summary_rows: list[dict] = []
    for label, kind, width, tcfg in specs:
        fold_test_means: list[dict] = []
        all_run_tests: list[dict] = []

        for f_idx, fold_data in enumerate(folds):
            if kind == "linear":
                lin = LinearBaseline().fit(fold_data.X_train, fold_data.y_train)
                tm = evaluate_model_physical(lin, fold_data.X_test, fold_data.y_test, fold_data)
                fold_test_means.append(tm)
                all_run_tests.append(tm)
            else:
                seed_tests = []
                for seed in seeds:
                    mlp = NeuralNetworkNumPy(input_dim=fold_data.n_features, hidden_width=width, activation="tanh", seed=seed)
                    train_model(mlp, fold_data.X_train, fold_data.y_train, fold_data.X_val, fold_data.y_val, tcfg)
                    tm = evaluate_model_physical(mlp, fold_data.X_test, fold_data.y_test, fold_data)
                    seed_tests.append(tm)
                    all_run_tests.append(tm)
                # Fold average over init seeds -> measures across-fold split variance
                fold_avg = {
                    k: float(np.mean([st[k] for st in seed_tests]))
                    for k in seed_tests[0]
                }
                fold_test_means.append(fold_avg)

        fold_agg = aggregate_seed_metrics(fold_test_means)
        total_agg = aggregate_seed_metrics(all_run_tests)
        summary_rows.append({
            "model": label,
            "n_folds": 5,
            "n_seeds_per_fold": 1 if kind == "linear" else len(seeds),
            "fold_means": fold_test_means,
            "across_fold_aggregate": fold_agg,
            "overall_run_aggregate": total_agg,
        })
        print(
            f"  {label:<28s} | 5-Fold RMSE={fold_agg['rmse_mean']:.3f}±{fold_agg['rmse_std']:.3f} MPa | "
            f"5-Fold R2={fold_agg['r2_mean']:.4f}±{fold_agg['r2_std']:.4f}"
        )

    return {"experiment": "E9C_concrete_5fold_cv", "results": summary_rows}


def run_e9d_double_descent() -> dict:
    """E9D: Extreme overparameterization H=4..1024 (P/N up to 16.57) + 2-HL depth comparison."""
    print("\n[E9D] Extreme overparameterization & 2-HL depth curve on Concrete (PyTorch float64) ...")
    rd = load_concrete_dataset(cfg=RealDataConfig())
    n_train = len(rd.X_train)
    seeds = INIT_SEEDS_REAL

    width_rows: list[dict] = []
    for H in EXTENDED_WIDTH_GRID:
        tr_list, val_list, test_list, gap_list = [], [], [], []
        n_params = 0
        for seed in seeds:
            model = NeuralNetworkTorch(input_dim=rd.n_features, hidden_width=H, activation="tanh")
            model.init_matching_numpy(seed)
            n_params = model.count_parameters()
            train_torch_mlp(model, rd.X_train, rd.y_train, lr=0.01, max_epochs=3000)

            tr_m = evaluate_model_physical(model, rd.X_train, rd.y_train, rd)
            val_m = evaluate_model_physical(model, rd.X_val, rd.y_val, rd)
            test_m = evaluate_model_physical(model, rd.X_test, rd.y_test, rd)
            tr_list.append(tr_m)
            val_list.append(val_m)
            test_list.append(test_m)
            gap_list.append(compute_generalization_gap(tr_m, val_m))

        tr_agg = aggregate_seed_metrics(tr_list)
        val_agg = aggregate_seed_metrics(val_list)
        test_agg = aggregate_seed_metrics(test_list)
        gap_agg = aggregate_seed_metrics(gap_list)
        pn = n_params / n_train
        width_rows.append({
            "hidden_width": H,
            "n_params": n_params,
            "p_over_n": float(pn),
            "train_aggregate": tr_agg,
            "val_aggregate": val_agg,
            "test_aggregate": test_agg,
            "gap_aggregate": gap_agg,
        })
        print(
            f"  1-HL H={H:>4d} (P={n_params:>5d}, P/N={pn:5.2f}) | "
            f"train={tr_agg['mse_mean']:5.2f} | val={val_agg['mse_mean']:5.2f}±{val_agg['mse_std']:4.2f} | "
            f"test={test_agg['mse_mean']:5.2f} (R2={test_agg['r2_mean']:.4f})"
        )

    # Depth comparison: 2-hidden-layer architectures matched approximately to P=641, 2561, 10241
    depth_rows: list[dict] = []
    for h1, h2 in [(20, 20), (45, 45), (95, 95)]:
        tr_list, val_list, test_list = [], [], []
        n_params = 0
        for seed in seeds:
            model = TwoHiddenLayerTorch(input_dim=rd.n_features, h1=h1, h2=h2, activation="tanh", seed=seed)
            n_params = model.count_parameters()
            train_torch_mlp(model, rd.X_train, rd.y_train, lr=0.01, max_epochs=3000)
            tr_list.append(evaluate_model_physical(model, rd.X_train, rd.y_train, rd))
            val_list.append(evaluate_model_physical(model, rd.X_val, rd.y_val, rd))
            test_list.append(evaluate_model_physical(model, rd.X_test, rd.y_test, rd))

        tr_agg = aggregate_seed_metrics(tr_list)
        val_agg = aggregate_seed_metrics(val_list)
        test_agg = aggregate_seed_metrics(test_list)
        pn = n_params / n_train
        depth_rows.append({
            "arch": f"2-HL [{h1},{h2}]",
            "h1": h1,
            "h2": h2,
            "n_params": n_params,
            "p_over_n": float(pn),
            "train_aggregate": tr_agg,
            "val_aggregate": val_agg,
            "test_aggregate": test_agg,
        })
        print(
            f"  2-HL [{h1:>2d},{h2:>2d}] (P={n_params:>5d}, P/N={pn:5.2f}) | "
            f"train={tr_agg['mse_mean']:5.2f} | val={val_agg['mse_mean']:5.2f} | "
            f"test={test_agg['mse_mean']:5.2f} (R2={test_agg['r2_mean']:.4f})"
        )

    return {
        "experiment": "E9D_double_descent_depth",
        "n_train": n_train,
        "one_hidden_layer_results": width_rows,
        "two_hidden_layer_results": depth_rows,
    }


def run_experiment() -> dict:
    t0 = time.perf_counter()
    paths = ProjectPaths()
    paths.ensure_dirs()

    payload = {
        "experiment": "E9_scientific_extensions",
        "E9A_warmup": run_e9a_warmup_ablation(),
        "E9B_minibatch": run_e9b_minibatch_sgd(),
        "E9C_kfold": run_e9c_concrete_kfold(),
        "E9D_double_descent": run_e9d_double_descent(),
    }

    out_path = paths.results_logs / "E9_extensions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2)
    print(f"\n[E9] All extensions saved -> {out_path} ({time.perf_counter() - t0:.1f}s)")
    return payload


if __name__ == "__main__":
    run_experiment()

