"""E5: Regularization strategies, weight vs bias penalty, coupled L2 vs AdamW, and early stopping.

Strict model-selection protocol:
  - Compares unregularized, coupled L2, bias-regularized L2 ablation, early
    stopping, and decoupled AdamW using ONLY train & validation sets.
  - Selects best strategy by lowest mean validation MSE, then evaluates test set.
"""

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
    WEIGHT_DECAY_GRID,
)
from src.data.real_data import load_concrete_dataset
from src.data.synthetic_data import build_synthetic_dataset
from src.evaluation.metrics import (
    aggregate_seed_metrics,
    compute_generalization_gap,
    evaluate_model_physical,
    to_serializable,
)
from src.models.neural_network_numpy import NeuralNetworkNumPy
from src.training.regularization import compute_weight_l2_norm_sq
from src.training.trainer import train_model


def _build_reg_configs() -> list[dict]:
    cfgs: list[dict] = [
        {
            "label": "unreg",
            "group": "unregularized",
            "strength": 0.0,
            "cfg": TrainConfig(optimizer="adam", learning_rate=0.01, max_epochs=2500, eval_val_every=5),
        }
    ]
    for lam in WEIGHT_DECAY_GRID:
        if lam == 0.0:
            continue
        cfgs.append({
            "label": f"l2_lambda={lam}",
            "group": "coupled_l2",
            "strength": lam,
            "cfg": TrainConfig(
                optimizer="adam", learning_rate=0.01, max_epochs=2500,
                l2_lambda=lam, regularize_biases=False, eval_val_every=5,
            ),
        })
    cfgs.append({
        "label": "l2_lambda=0.01_with_bias",
        "group": "bias_ablation",
        "strength": 0.01,
        "cfg": TrainConfig(
            optimizer="adam", learning_rate=0.01, max_epochs=2500,
            l2_lambda=0.01, regularize_biases=True, eval_val_every=5,
        ),
    })
    cfgs.append({
        "label": "early_stop",
        "group": "early_stopping",
        "strength": 0.0,
        "cfg": TrainConfig(
            optimizer="adam", learning_rate=0.01, max_epochs=2500,
            early_stopping=True, patience=60, min_delta=1e-5, restore_best=True,
        ),
    })
    for wd in WEIGHT_DECAY_GRID:
        if wd == 0.0:
            continue
        cfgs.append({
            "label": f"adamw_wd={wd}",
            "group": "decoupled_adamw",
            "strength": wd,
            "cfg": TrainConfig(
                optimizer="adamw", learning_rate=0.01, max_epochs=2500,
                weight_decay=wd, regularize_biases=False, eval_val_every=5,
            ),
        })
    return cfgs


def _run_reg_sweep(dataset_name: str, data, input_dim: int, seeds: list[int], reg_cfgs: list[dict]) -> dict:
    overparam_width = 128
    all_entries: list[dict] = []
    trained_by_label: dict[str, list[NeuralNetworkNumPy]] = {}

    for rc in reg_cfgs:
        label = rc["label"]
        cfg = rc["cfg"]
        seed_records: list[dict] = []
        models: list[NeuralNetworkNumPy] = []

        for idx, seed in enumerate(seeds):
            mlp = NeuralNetworkNumPy(
                input_dim=input_dim,
                hidden_width=overparam_width,
                activation="tanh",
                seed=seed,
            )
            hist = train_model(
                mlp, data.X_train, data.y_train, data.X_val, data.y_val, cfg=cfg
            )
            tr_m = evaluate_model_physical(mlp, data.X_train, data.y_train, data)
            val_m = evaluate_model_physical(mlp, data.X_val, data.y_val, data)
            gap = compute_generalization_gap(tr_m, val_m)
            w_norm_sq = compute_weight_l2_norm_sq(mlp.W1, mlp.W2)

            models.append(mlp)
            seed_records.append({
                "seed": seed,
                "epochs_run": hist.epochs_run,
                "early_stopped": hist.early_stopped,
                "stopped_epoch": hist.stopped_epoch,
                "best_val_epoch": hist.best_val_epoch,
                "weight_l2_norm_sq": w_norm_sq,
                "train_metrics": tr_m,
                "val_metrics": val_m,
                "gen_gap": gap,
                "train_mse_curve": hist.train_mse if idx == 0 else None,
                "val_mse_curve": hist.val_mse if idx == 0 else None,
            })

        trained_by_label[label] = models
        train_agg = aggregate_seed_metrics([r["train_metrics"] for r in seed_records])
        val_agg = aggregate_seed_metrics([r["val_metrics"] for r in seed_records])
        gap_agg = aggregate_seed_metrics([r["gen_gap"] for r in seed_records])
        norm_agg = aggregate_seed_metrics([{"weight_l2_norm_sq": r["weight_l2_norm_sq"]} for r in seed_records])

        all_entries.append({
            "label": label,
            "group": rc["group"],
            "strength": rc["strength"],
            "config": cfg.to_dict(),
            "seed_runs": seed_records,
            "train_aggregate": train_agg,
            "val_aggregate": val_agg,
            "gap_aggregate": gap_agg,
            "weight_norm_aggregate": norm_agg,
        })
        print(
            f"  [{dataset_name}] {label:<24s} | train={train_agg['mse_mean']:.4f} | "
            f"val={val_agg['mse_mean']:.4f}±{val_agg['mse_std']:.4f} | "
            f"||W||^2={norm_agg['weight_l2_norm_sq_mean']:.1f}"
        )

    # Validation-only model selection
    best_overall = min(all_entries, key=lambda e: e["val_aggregate"]["mse_mean"])
    l2_entries = [e for e in all_entries if e["group"] == "coupled_l2"]
    best_l2 = min(l2_entries, key=lambda e: e["val_aggregate"]["mse_mean"])
    wd_entries = [e for e in all_entries if e["group"] == "decoupled_adamw"]
    best_adamw = min(wd_entries, key=lambda e: e["val_aggregate"]["mse_mean"])

    # Post-selection test evaluation on final representative strategies
    eval_labels = list(dict.fromkeys([
        "unreg",
        "early_stop",
        best_l2["label"],
        best_adamw["label"],
        best_overall["label"],
    ]))
    post_selection_test: dict[str, dict] = {}
    for lbl in eval_labels:
        test_runs = [
            evaluate_model_physical(m, data.X_test, data.y_test, data)
            for m in trained_by_label[lbl]
        ]
        post_selection_test[lbl] = aggregate_seed_metrics(test_runs)

    print(
        f"  --> Selected best overall='{best_overall['label']}' (val_mse={best_overall['val_aggregate']['mse_mean']:.4f}); "
        f"test_mse={post_selection_test[best_overall['label']]['mse_mean']:.4f} "
        f"(test_R2={post_selection_test[best_overall['label']]['r2_mean']:.4f})"
    )

    return {
        "dataset": dataset_name,
        "hidden_width": overparam_width,
        "reg_results": all_entries,
        "best_overall_label": best_overall["label"],
        "best_l2_label": best_l2["label"],
        "best_adamw_label": best_adamw["label"],
        "post_selection_test": post_selection_test,
    }


def run_experiment() -> dict:
    paths = ProjectPaths()
    paths.ensure_dirs()
    reg_cfgs = _build_reg_configs()

    print("[E5] Synthetic regularization sweep ...")
    syn_data = build_synthetic_dataset(cfg=SyntheticDataConfig())
    syn_res = _run_reg_sweep("synthetic", syn_data, 1, INIT_SEEDS_SYN[:5], reg_cfgs)
    out_syn = paths.results_logs / "E5_regularization_synthetic.json"
    with open(out_syn, "w", encoding="utf-8") as f:
        json.dump(to_serializable({"experiment": "E5_regularization_synthetic", **syn_res}), f, indent=2)
    print(f"  Saved -> {out_syn}")

    print("[E5] Concrete regularization sweep ...")
    real_data = load_concrete_dataset(cfg=RealDataConfig())
    real_res = _run_reg_sweep("concrete", real_data, real_data.n_features, INIT_SEEDS_REAL, reg_cfgs)
    out_real = paths.results_logs / "E5_regularization_concrete.json"
    with open(out_real, "w", encoding="utf-8") as f:
        json.dump(to_serializable({"experiment": "E5_regularization_concrete", **real_res}), f, indent=2)
    print(f"  Saved -> {out_real}")

    return {"experiment": "E5_regularization", "synthetic": syn_res, "concrete": real_res}


if __name__ == "__main__":
    run_experiment()
