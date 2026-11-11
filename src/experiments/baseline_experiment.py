"""E2: Linear baseline vs one-hidden-layer MLP pilot experiment."""

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
from src.data.synthetic_data import build_synthetic_dataset, ground_truth_function
from src.evaluation.metrics import (
    aggregate_seed_metrics,
    compute_generalization_gap,
    evaluate_model_physical,
    to_serializable,
)
from src.models.linear_baseline import LinearBaseline
from src.models.neural_network_numpy import NeuralNetworkNumPy
from src.training.trainer import train_model


def run_experiment() -> dict:
    paths = ProjectPaths()
    paths.ensure_dirs()

    syn_cfg = SyntheticDataConfig()
    real_cfg = RealDataConfig()
    train_cfg = TrainConfig(
        optimizer="adam",
        learning_rate=0.01,
        batch_size=None,
        max_epochs=2500,
    )

    results: dict = {"experiment": "E2_baseline", "runs": [], "aggregates": {}, "plot_data": {}}

    syn_data = build_synthetic_dataset(cfg=syn_cfg)
    real_data = load_concrete_dataset(cfg=real_cfg)

    datasets = [
        ("synthetic", syn_data, 1, INIT_SEEDS_SYN[:5]),
        ("concrete", real_data, real_data.n_features, INIT_SEEDS_REAL),
    ]

    for dname, data, input_dim, seeds in datasets:
        # Linear baseline (deterministic)
        lin = LinearBaseline().fit(data.X_train, data.y_train)
        lin_tr = evaluate_model_physical(lin, data.X_train, data.y_train, data)
        lin_val = evaluate_model_physical(lin, data.X_val, data.y_val, data)
        lin_test = evaluate_model_physical(lin, data.X_test, data.y_test, data)
        results["runs"].append({
            "dataset": dname,
            "model": "linear",
            "activation": None,
            "seed": 0,
            "n_params": lin.count_parameters(),
            "train_metrics": lin_tr,
            "val_metrics": lin_val,
            "test_metrics": lin_test,
            "gen_gap": compute_generalization_gap(lin_tr, lin_val),
        })
        results["aggregates"][f"{dname}_linear"] = {
            "train": aggregate_seed_metrics([lin_tr]),
            "val": aggregate_seed_metrics([lin_val]),
            "test": aggregate_seed_metrics([lin_test]),
        }

        for act in ("tanh", "relu"):
            tr_list, val_list, test_list = [], [], []
            rep_model = None
            for idx, seed in enumerate(seeds):
                mlp = NeuralNetworkNumPy(
                    input_dim=input_dim,
                    hidden_width=32,
                    activation=act,
                    seed=seed,
                )
                hist = train_model(
                    mlp, data.X_train, data.y_train, data.X_val, data.y_val, cfg=train_cfg
                )
                tr_m = evaluate_model_physical(mlp, data.X_train, data.y_train, data)
                val_m = evaluate_model_physical(mlp, data.X_val, data.y_val, data)
                test_m = evaluate_model_physical(mlp, data.X_test, data.y_test, data)

                tr_list.append(tr_m)
                val_list.append(val_m)
                test_list.append(test_m)
                if idx == 0:
                    rep_model = mlp

                results["runs"].append({
                    "dataset": dname,
                    "model": "mlp",
                    "activation": act,
                    "hidden_width": 32,
                    "seed": seed,
                    "n_params": mlp.count_parameters(),
                    "epochs_run": hist.epochs_run,
                    "train_metrics": tr_m,
                    "val_metrics": val_m,
                    "test_metrics": test_m,
                    "gen_gap": compute_generalization_gap(tr_m, val_m),
                })

            results["aggregates"][f"{dname}_mlp_{act}"] = {
                "train": aggregate_seed_metrics(tr_list),
                "val": aggregate_seed_metrics(val_list),
                "test": aggregate_seed_metrics(test_list),
            }

            # Save representative predictions for plotting
            if dname == "synthetic" and act == "tanh" and rep_model is not None:
                x_grid = np.linspace(syn_cfg.x_min, syn_cfg.x_max, 300).reshape(-1, 1)
                results["plot_data"]["synthetic"] = {
                    "X_train": data.X_train.ravel().tolist(),
                    "y_train": data.y_train.ravel().tolist(),
                    "X_plot": x_grid.ravel().tolist(),
                    "f_true": ground_truth_function(x_grid).ravel().tolist(),
                    "linear_pred": lin.predict(x_grid).ravel().tolist(),
                    "nn_tanh_pred": rep_model.predict(x_grid).ravel().tolist(),
                }
            elif dname == "synthetic" and act == "relu" and rep_model is not None:
                x_grid = np.linspace(syn_cfg.x_min, syn_cfg.x_max, 300).reshape(-1, 1)
                results["plot_data"]["synthetic"]["nn_relu_pred"] = rep_model.predict(x_grid).ravel().tolist()
            elif dname == "concrete" and act == "tanh" and rep_model is not None:
                results["plot_data"]["concrete"] = {
                    "y_test_mpa": data.inverse_transform_y(data.y_test).ravel().tolist(),
                    "linear_pred_mpa": data.inverse_transform_y(lin.predict(data.X_test)).ravel().tolist(),
                    "nn_tanh_pred_mpa": data.inverse_transform_y(rep_model.predict(data.X_test)).ravel().tolist(),
                }

    out_path = paths.results_logs / "E2_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"[E2] Saved -> {out_path}")

    print("\n" + "=" * 92)
    print(f"{'Config':<22} {'Train MSE':>11} {'Val MSE':>11} {'Test MSE':>11} {'Test RMSE':>11} {'Test R2':>9}")
    print("-" * 92)
    for key, agg in results["aggregates"].items():
        tr = agg["train"]
        va = agg["val"]
        te = agg["test"]
        print(
            f"{key:<22} {tr['mse_mean']:>11.4f} {va['mse_mean']:>11.4f} "
            f"{te['mse_mean']:>11.4f} {te['rmse_mean']:>11.4f} {te['r2_mean']:>9.4f}"
        )
    print("=" * 92)
    return results


if __name__ == "__main__":
    run_experiment()
