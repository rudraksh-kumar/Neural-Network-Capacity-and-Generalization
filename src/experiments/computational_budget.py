"""E7 + E8: Computational budget protocols (A & B) and controlled diagnostics."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from config.experiment_config import (
    INIT_SEEDS_REAL,
    INIT_SEEDS_SYN,
    ProjectPaths,
    RealDataConfig,
    SyntheticDataConfig,
    TrainConfig,
)
from src.data.real_data import load_concrete_dataset
from src.data.synthetic_data import build_synthetic_dataset
from src.evaluation.metrics import (
    aggregate_seed_metrics,
    compute_generalization_gap,
    evaluate_model_physical,
    to_serializable,
)
from src.models.linear_baseline import LinearBaseline
from src.models.neural_network_numpy import NeuralNetworkNumPy
from src.training.trainer import train_model


BUDGET_WIDTHS = [4, 16, 32, 64, 128, 256]
FIXED_STEP_BUDGET = 2000
LR_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
OPTIMIZER_LIST = ["gd", "momentum", "adam"]


def _train_and_record(mlp: NeuralNetworkNumPy, data, cfg: TrainConfig) -> dict:
    hist = train_model(mlp, data.X_train, data.y_train, data.X_val, data.y_val, cfg=cfg)
    tr_m = evaluate_model_physical(mlp, data.X_train, data.y_train, data)
    val_m = evaluate_model_physical(mlp, data.X_val, data.y_val, data)
    gap = compute_generalization_gap(tr_m, val_m)
    return {
        "n_params": mlp.count_parameters(),
        "epochs_run": hist.epochs_run,
        "steps_run": hist.steps_run,
        "examples_processed": int(hist.steps_run * len(data.X_train)),
        "total_flops": hist.total_flops,
        "wall_time_sec": hist.wall_time_sec,
        "converged": hist.converged,
        "budget_exhausted": hist.budget_exhausted,
        "train_metrics": tr_m,
        "val_metrics": val_m,
        "gen_gap": gap,
    }


def _run_protocol_a(dataset_name: str, data, input_dim: int, seeds: list[int]) -> dict:
    """Protocol A: Fixed step budget AND matched FLOPs budget across widths."""
    fixed_step_rows: list[dict] = []
    matched_flop_rows: list[dict] = []

    ref_params = NeuralNetworkNumPy(input_dim=input_dim, hidden_width=32, activation="tanh").count_parameters()
    target_flops = float(FIXED_STEP_BUDGET * 6 * len(data.X_train) * ref_params)

    for H in BUDGET_WIDTHS:
        # 1. Fixed step budget (2000 steps)
        cfg_step = TrainConfig(
            optimizer="adam", learning_rate=0.01, max_epochs=5000,
            max_steps=FIXED_STEP_BUDGET, eval_val_every=20,
        )
        step_runs = []
        for seed in seeds:
            mlp = NeuralNetworkNumPy(input_dim=input_dim, hidden_width=H, activation="tanh", seed=seed)
            rec = _train_and_record(mlp, data, cfg_step)
            rec["seed"] = seed
            step_runs.append(rec)

        fixed_step_rows.append({
            "hidden_width": H,
            "n_params": step_runs[0]["n_params"],
            "steps_run": step_runs[0]["steps_run"],
            "examples_processed": step_runs[0]["examples_processed"],
            "total_flops": step_runs[0]["total_flops"],
            "mean_wall_time_sec": float(np.mean([r["wall_time_sec"] for r in step_runs])),
            "train_aggregate": aggregate_seed_metrics([r["train_metrics"] for r in step_runs]),
            "val_aggregate": aggregate_seed_metrics([r["val_metrics"] for r in step_runs]),
        })

        # 2. Matched FLOPs budget (equalized forward+backward MAC budget)
        cfg_flop = TrainConfig(
            optimizer="adam", learning_rate=0.01, max_epochs=50000,
            max_flops=target_flops, eval_val_every=20,
        )
        flop_runs = []
        for seed in seeds:
            mlp = NeuralNetworkNumPy(input_dim=input_dim, hidden_width=H, activation="tanh", seed=seed)
            rec = _train_and_record(mlp, data, cfg_flop)
            rec["seed"] = seed
            flop_runs.append(rec)

        matched_flop_rows.append({
            "hidden_width": H,
            "n_params": flop_runs[0]["n_params"],
            "steps_run": flop_runs[0]["steps_run"],
            "examples_processed": flop_runs[0]["examples_processed"],
            "total_flops": flop_runs[0]["total_flops"],
            "mean_wall_time_sec": float(np.mean([r["wall_time_sec"] for r in flop_runs])),
            "train_aggregate": aggregate_seed_metrics([r["train_metrics"] for r in flop_runs]),
            "val_aggregate": aggregate_seed_metrics([r["val_metrics"] for r in flop_runs]),
        })
        print(
            f"  [E7-A {dataset_name[:3]}] H={H:>3d} | "
            f"fixed_step({fixed_step_rows[-1]['steps_run']}st) val={fixed_step_rows[-1]['val_aggregate']['mse_mean']:.4f} | "
            f"matched_flop({matched_flop_rows[-1]['steps_run']}st) val={matched_flop_rows[-1]['val_aggregate']['mse_mean']:.4f}"
        )

    return {
        "protocol": "A_common_budget",
        "fixed_step_budget": FIXED_STEP_BUDGET,
        "matched_flops_budget": target_flops,
        "fixed_step_results": fixed_step_rows,
        "matched_flop_results": matched_flop_rows,
    }


def _run_protocol_b(dataset_name: str, data, input_dim: int, seeds: list[int]) -> dict:
    """Protocol B: Convergence criterion (val delta < 1e-5 for 50 epochs) + max 5000 epochs."""
    cfg = TrainConfig(
        optimizer="adam",
        learning_rate=0.01,
        max_epochs=5000,
        convergence_tol=1e-5,
        convergence_patience=50,
    )
    rows: list[dict] = []
    for H in BUDGET_WIDTHS:
        runs = []
        for seed in seeds:
            mlp = NeuralNetworkNumPy(input_dim=input_dim, hidden_width=H, activation="tanh", seed=seed)
            rec = _train_and_record(mlp, data, cfg)
            rec["seed"] = seed
            runs.append(rec)

        rows.append({
            "hidden_width": H,
            "n_params": runs[0]["n_params"],
            "mean_epochs": float(np.mean([r["epochs_run"] for r in runs])),
            "mean_steps": float(np.mean([r["steps_run"] for r in runs])),
            "mean_flops": float(np.mean([r["total_flops"] for r in runs])),
            "mean_wall_time_sec": float(np.mean([r["wall_time_sec"] for r in runs])),
            "convergence_rate": float(np.mean([1.0 if r["converged"] else 0.0 for r in runs])),
            "train_aggregate": aggregate_seed_metrics([r["train_metrics"] for r in runs]),
            "val_aggregate": aggregate_seed_metrics([r["val_metrics"] for r in runs]),
        })
        print(
            f"  [E7-B {dataset_name[:3]}] H={H:>3d} | epochs={rows[-1]['mean_epochs']:.0f} | "
            f"conv_rate={rows[-1]['convergence_rate']:.2f} | val={rows[-1]['val_aggregate']['mse_mean']:.4f}"
        )

    return {
        "protocol": "B_convergence",
        "convergence_tol": 1e-5,
        "convergence_patience": 50,
        "max_epochs": 5000,
        "results": rows,
    }


def _run_diagnostics(dataset_name: str, data, input_dim: int, seeds: list[int]) -> dict:
    """E8: Controlled interventions for optimization failure, overfitting, inadequate rep."""
    # 1. LR sweep (optimization failure check)
    lr_rows = []
    for lr in LR_GRID:
        cfg = TrainConfig(optimizer="adam", learning_rate=lr, max_epochs=2500, eval_val_every=20)
        runs = [_train_and_record(NeuralNetworkNumPy(input_dim, 32, "tanh", seed=s), data, cfg) for s in seeds]
        lr_rows.append({
            "lr": lr,
            "train_aggregate": aggregate_seed_metrics([r["train_metrics"] for r in runs]),
            "val_aggregate": aggregate_seed_metrics([r["val_metrics"] for r in runs]),
        })

    # 2. Optimizer sweep
    opt_rows = []
    for opt_name in OPTIMIZER_LIST:
        cfg = TrainConfig(optimizer=opt_name, learning_rate=0.01, max_epochs=2500, eval_val_every=20)
        runs = [_train_and_record(NeuralNetworkNumPy(input_dim, 32, "tanh", seed=s), data, cfg) for s in seeds]
        opt_rows.append({
            "optimizer": opt_name,
            "train_aggregate": aggregate_seed_metrics([r["train_metrics"] for r in runs]),
            "val_aggregate": aggregate_seed_metrics([r["val_metrics"] for r in runs]),
        })

    # 3. Initialization scale sweep
    init_rows = []
    for init_label, scheme, scale in [
        ("tiny_0.001", "normal", 0.001),
        ("xavier_auto", "auto", None),
        ("large_3.0", "normal", 3.0),
    ]:
        cfg = TrainConfig(optimizer="adam", learning_rate=0.01, max_epochs=2500, eval_val_every=20)
        runs = [
            _train_and_record(
                NeuralNetworkNumPy(input_dim, 32, "tanh", init_scheme=scheme, init_scale=scale, seed=s),
                data,
                cfg,
            )
            for s in seeds
        ]
        init_rows.append({
            "init_label": init_label,
            "train_aggregate": aggregate_seed_metrics([r["train_metrics"] for r in runs]),
            "val_aggregate": aggregate_seed_metrics([r["val_metrics"] for r in runs]),
        })

    # 4. Overfitting controlled interventions (small N, H=128)
    small_data = (
        build_synthetic_dataset(cfg=SyntheticDataConfig(), n_train=50)
        if dataset_name == "synthetic"
        else load_concrete_dataset(cfg=RealDataConfig(), n_train_subsample=150)
    )
    large_data = (
        build_synthetic_dataset(cfg=SyntheticDataConfig(), n_train=400)
        if dataset_name == "synthetic"
        else data
    )
    ov_rows = []
    for lbl, ds_use, cfg in [
        ("unreg_H128_smallN", small_data, TrainConfig(optimizer="adam", learning_rate=0.01, max_epochs=2500, eval_val_every=20)),
        ("l2_0.01_H128_smallN", small_data, TrainConfig(optimizer="adam", learning_rate=0.01, max_epochs=2500, l2_lambda=0.01, eval_val_every=20)),
        ("early_stop_H128_smallN", small_data, TrainConfig(optimizer="adam", learning_rate=0.01, max_epochs=2500, early_stopping=True, patience=50, min_delta=1e-5)),
        ("larger_N_H128", large_data, TrainConfig(optimizer="adam", learning_rate=0.01, max_epochs=2500, eval_val_every=20)),
    ]:
        runs = [_train_and_record(NeuralNetworkNumPy(input_dim, 128, "tanh", seed=s), ds_use, cfg) for s in seeds]
        ov_rows.append({
            "label": lbl,
            "train_aggregate": aggregate_seed_metrics([r["train_metrics"] for r in runs]),
            "val_aggregate": aggregate_seed_metrics([r["val_metrics"] for r in runs]),
        })

    # 5. Inadequate representation interventions
    lin = LinearBaseline().fit(data.X_train, data.y_train)
    rep_rows = [
        {
            "label": "linear_baseline",
            "train_aggregate": aggregate_seed_metrics([evaluate_model_physical(lin, data.X_train, data.y_train, data)]),
            "val_aggregate": aggregate_seed_metrics([evaluate_model_physical(lin, data.X_val, data.y_val, data)]),
        }
    ]
    for lbl, width, act in [
        ("mlp_H=2_tanh", 2, "tanh"),
        ("mlp_H=4_tanh", 4, "tanh"),
        ("mlp_H=32_tanh", 32, "tanh"),
        ("mlp_H=32_relu", 32, "relu"),
    ]:
        cfg = TrainConfig(optimizer="adam", learning_rate=0.01, max_epochs=2500, eval_val_every=20)
        runs = [_train_and_record(NeuralNetworkNumPy(input_dim, width, act, seed=s), data, cfg) for s in seeds]
        rep_rows.append({
            "label": lbl,
            "train_aggregate": aggregate_seed_metrics([r["train_metrics"] for r in runs]),
            "val_aggregate": aggregate_seed_metrics([r["val_metrics"] for r in runs]),
        })

    return {
        "dataset": dataset_name,
        "lr_sweep": {"lr_grid": LR_GRID, "results": lr_rows},
        "optimizer_sweep": {"optimizers": OPTIMIZER_LIST, "results": opt_rows},
        "init_sweep": {"results": init_rows},
        "overfitting_interventions": {"results": ov_rows},
        "representation_interventions": {"results": rep_rows},
    }


def run_experiment() -> dict:
    paths = ProjectPaths()
    paths.ensure_dirs()

    syn_data = build_synthetic_dataset(cfg=SyntheticDataConfig())
    real_data = load_concrete_dataset(cfg=RealDataConfig())

    print("[E7] Running Protocol A & B ...")
    e7_syn = {
        "protocol_a": _run_protocol_a("synthetic", syn_data, 1, INIT_SEEDS_SYN[:5]),
        "protocol_b": _run_protocol_b("synthetic", syn_data, 1, INIT_SEEDS_SYN[:5]),
    }
    e7_real = {
        "protocol_a": _run_protocol_a("concrete", real_data, real_data.n_features, INIT_SEEDS_REAL),
        "protocol_b": _run_protocol_b("concrete", real_data, real_data.n_features, INIT_SEEDS_REAL),
    }

    for fname, payload in [("E7_budget_synthetic.json", e7_syn), ("E7_budget_concrete.json", e7_real)]:
        with open(paths.results_logs / fname, "w", encoding="utf-8") as f:
            json.dump(to_serializable({"experiment": fname.replace(".json", ""), **payload}), f, indent=2)

    print("[E8] Running controlled diagnostic interventions ...")
    e8_syn = _run_diagnostics("synthetic", syn_data, 1, INIT_SEEDS_SYN[:5])
    e8_real = _run_diagnostics("concrete", real_data, real_data.n_features, INIT_SEEDS_REAL)

    for fname, payload in [("E8_diagnostics_synthetic.json", e8_syn), ("E8_diagnostics_concrete.json", e8_real)]:
        with open(paths.results_logs / fname, "w", encoding="utf-8") as f:
            json.dump(to_serializable({"experiment": fname.replace(".json", ""), **payload}), f, indent=2)

    print("[E7+E8] Completed and saved all budget & diagnostic logs.")
    return {"E7_synthetic": e7_syn, "E7_concrete": e7_real, "E8_synthetic": e8_syn, "E8_concrete": e8_real}


if __name__ == "__main__":
    run_experiment()
