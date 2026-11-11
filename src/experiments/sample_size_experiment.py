"""E4: Controlled training-set size sweep with fixed hidden width (H=32)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from config.experiment_config import (
    INIT_SEEDS_REAL,
    INIT_SEEDS_SYN,
    ProjectPaths,
    RealDataConfig,
    SAMPLE_SIZE_GRID_REAL,
    SAMPLE_SIZE_GRID_SYN,
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


def _sweep_dataset(dataset_name: str, n_grid: list[int], seeds: list[int], train_cfg: TrainConfig) -> dict:
    syn_cfg = SyntheticDataConfig()
    real_cfg = RealDataConfig()
    size_results: list[dict] = []
    fixed_width = 32

    for N in n_grid:
        seed_records: list[dict] = []
        lin_val_records: list[dict] = []

        for seed in seeds:
            if dataset_name == "synthetic":
                data = build_synthetic_dataset(cfg=syn_cfg, n_train=N, train_seed=1000 + seed)
                input_dim = 1
            else:
                data = load_concrete_dataset(cfg=real_cfg, n_train_subsample=N, subsample_seed=700 + seed)
                input_dim = data.n_features

            lin = LinearBaseline().fit(data.X_train, data.y_train)
            lin_val_records.append(evaluate_model_physical(lin, data.X_val, data.y_val, data))

            mlp = NeuralNetworkNumPy(
                input_dim=input_dim,
                hidden_width=fixed_width,
                activation="tanh",
                seed=seed,
            )
            hist = train_model(
                mlp, data.X_train, data.y_train, data.X_val, data.y_val, cfg=train_cfg
            )
            tr_m = evaluate_model_physical(mlp, data.X_train, data.y_train, data)
            val_m = evaluate_model_physical(mlp, data.X_val, data.y_val, data)
            test_m = evaluate_model_physical(mlp, data.X_test, data.y_test, data)
            gap = compute_generalization_gap(tr_m, val_m)

            seed_records.append({
                "seed": seed,
                "n_train": N,
                "n_params": mlp.count_parameters(),
                "epochs_run": hist.epochs_run,
                "train_metrics": tr_m,
                "val_metrics": val_m,
                "test_metrics": test_m,
                "gen_gap": gap,
            })

        train_agg = aggregate_seed_metrics([r["train_metrics"] for r in seed_records])
        val_agg = aggregate_seed_metrics([r["val_metrics"] for r in seed_records])
        test_agg = aggregate_seed_metrics([r["test_metrics"] for r in seed_records])
        gap_agg = aggregate_seed_metrics([r["gen_gap"] for r in seed_records])
        lin_val_agg = aggregate_seed_metrics(lin_val_records)

        size_results.append({
            "n_train": N,
            "n_params": seed_records[0]["n_params"],
            "seed_runs": seed_records,
            "train_aggregate": train_agg,
            "val_aggregate": val_agg,
            "test_aggregate": test_agg,
            "gap_aggregate": gap_agg,
            "linear_val_aggregate": lin_val_agg,
        })
        print(
            f"  [{dataset_name}] N={N:>4d} | train_mse={train_agg['mse_mean']:.4f}±{train_agg['mse_std']:.4f} | "
            f"val_mse={val_agg['mse_mean']:.4f}±{val_agg['mse_std']:.4f} | "
            f"gap={gap_agg['mse_gap_mean']:.4f} | lin_val={lin_val_agg['mse_mean']:.4f}"
        )

    return {
        "dataset": dataset_name,
        "fixed_width": fixed_width,
        "sample_sizes": n_grid,
        "irreducible_mse": syn_cfg.irreducible_mse if dataset_name == "synthetic" else None,
        "size_results": size_results,
    }


def run_experiment() -> dict:
    paths = ProjectPaths()
    paths.ensure_dirs()

    train_cfg = TrainConfig(
        optimizer="adam",
        learning_rate=0.01,
        batch_size=None,
        max_epochs=2500,
        eval_val_every=10,
    )

    print("[E4] Synthetic sample-size sweep ...")
    syn_res = _sweep_dataset("synthetic", SAMPLE_SIZE_GRID_SYN, INIT_SEEDS_SYN, train_cfg)
    out_syn = paths.results_logs / "E4_sample_size_synthetic.json"
    with open(out_syn, "w", encoding="utf-8") as f:
        json.dump(to_serializable({"experiment": "E4_sample_size_synthetic", **syn_res}), f, indent=2)
    print(f"  Saved -> {out_syn}")

    print("[E4] Concrete sample-size sweep ...")
    real_res = _sweep_dataset("concrete", SAMPLE_SIZE_GRID_REAL, INIT_SEEDS_REAL, train_cfg)
    out_real = paths.results_logs / "E4_sample_size_concrete.json"
    with open(out_real, "w", encoding="utf-8") as f:
        json.dump(to_serializable({"experiment": "E4_sample_size_concrete", **real_res}), f, indent=2)
    print(f"  Saved -> {out_real}")

    return {"experiment": "E4_sample_size", "synthetic": syn_res, "concrete": real_res}


if __name__ == "__main__":
    run_experiment()
