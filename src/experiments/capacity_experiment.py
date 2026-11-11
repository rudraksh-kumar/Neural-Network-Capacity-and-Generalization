"""E3: Hidden-layer width capacity sweep experiment.

Strict model-selection protocol:
  - Test set is NOT evaluated during the width sweep.
  - Best width H* is selected strictly by lowest mean validation MSE.
  - Only after selection are the selected H* models evaluated on the test set.
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
    WIDTH_GRID,
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


def _sweep_one_dataset(
    dataset_name: str,
    data,
    input_dim: int,
    seeds: list[int],
    train_cfg: TrainConfig,
) -> dict:
    """Sweep WIDTH_GRID using ONLY train & val sets, then evaluate best H* on test."""
    lin = LinearBaseline().fit(data.X_train, data.y_train)
    lin_tr = evaluate_model_physical(lin, data.X_train, data.y_train, data)
    lin_val = evaluate_model_physical(lin, data.X_val, data.y_val, data)

    width_results: list[dict] = []
    trained_models_by_width: dict[int, list[NeuralNetworkNumPy]] = {}
    representative_curves: dict[str, dict] = {}

    for H in WIDTH_GRID:
        seed_records: list[dict] = []
        models_for_h: list[NeuralNetworkNumPy] = []

        for idx, seed in enumerate(seeds):
            mlp = NeuralNetworkNumPy(
                input_dim=input_dim,
                hidden_width=H,
                activation="tanh",
                seed=seed,
            )
            history = train_model(
                mlp, data.X_train, data.y_train, data.X_val, data.y_val, cfg=train_cfg
            )
            train_m = evaluate_model_physical(mlp, data.X_train, data.y_train, data)
            val_m = evaluate_model_physical(mlp, data.X_val, data.y_val, data)
            gap = compute_generalization_gap(train_m, val_m)

            models_for_h.append(mlp)
            seed_records.append({
                "seed": seed,
                "n_params": mlp.count_parameters(),
                "epochs_run": history.epochs_run,
                "train_metrics": train_m,
                "val_metrics": val_m,
                "gen_gap": gap,
            })

            if idx == 0 and H in (4, 16, 32, 128, 256):
                representative_curves[f"H={H}"] = {
                    "train_mse": history.train_mse,
                    "val_mse": history.val_mse,
                }

        trained_models_by_width[H] = models_for_h
        train_agg = aggregate_seed_metrics([r["train_metrics"] for r in seed_records])
        val_agg = aggregate_seed_metrics([r["val_metrics"] for r in seed_records])
        gap_agg = aggregate_seed_metrics([r["gen_gap"] for r in seed_records])

        width_results.append({
            "hidden_width": H,
            "n_params": seed_records[0]["n_params"],
            "n_train": len(data.X_train),
            "seed_runs": seed_records,
            "train_aggregate": train_agg,
            "val_aggregate": val_agg,
            "gap_aggregate": gap_agg,
        })
        print(
            f"  [{dataset_name}] H={H:>4d} (P={seed_records[0]['n_params']:>4d}) | "
            f"train_mse={train_agg['mse_mean']:.4f}±{train_agg['mse_std']:.4f} | "
            f"val_mse={val_agg['mse_mean']:.4f}±{val_agg['mse_std']:.4f} | "
            f"gap={gap_agg['mse_gap_mean']:.4f}"
        )

    # Strict validation-only selection
    best_entry = min(width_results, key=lambda w: w["val_aggregate"]["mse_mean"])
    best_H = best_entry["hidden_width"]

    # Post-selection test evaluation (untouched test set opened ONLY here)
    best_test_runs = [
        evaluate_model_physical(m, data.X_test, data.y_test, data)
        for m in trained_models_by_width[best_H]
    ]
    selected_test_agg = aggregate_seed_metrics(best_test_runs)
    lin_test = evaluate_model_physical(lin, data.X_test, data.y_test, data)

    print(
        f"  --> Selected best_H={best_H} by val_mse={best_entry['val_aggregate']['mse_mean']:.4f}; "
        f"post-selection test_mse={selected_test_agg['mse_mean']:.4f}±{selected_test_agg['mse_std']:.4f} "
        f"(test_R2={selected_test_agg['r2_mean']:.4f})"
    )

    return {
        "dataset": dataset_name,
        "target_unit": data.target_unit,
        "irreducible_mse": data.irreducible_mse,
        "n_train": len(data.X_train),
        "input_dim": input_dim,
        "width_grid": WIDTH_GRID,
        "linear_baseline": {"train": lin_tr, "val": lin_val, "test": lin_test},
        "width_results": width_results,
        "best_H": best_H,
        "best_val_aggregate": best_entry["val_aggregate"],
        "selected_test_aggregate": selected_test_agg,
        "representative_curves": representative_curves,
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

    results: dict = {"experiment": "E3_capacity"}

    print("[E3] Synthetic width sweep ...")
    syn_data = build_synthetic_dataset(cfg=SyntheticDataConfig())
    syn_res = _sweep_one_dataset("synthetic", syn_data, 1, INIT_SEEDS_SYN, train_cfg)
    results["synthetic"] = syn_res

    out_syn = paths.results_logs / "E3_capacity_synthetic.json"
    with open(out_syn, "w", encoding="utf-8") as f:
        json.dump(to_serializable({"experiment": "E3_capacity_synthetic", **syn_res}), f, indent=2)
    print(f"  Saved -> {out_syn}")

    print("[E3] Concrete width sweep ...")
    real_data = load_concrete_dataset(cfg=RealDataConfig())
    real_res = _sweep_one_dataset("concrete", real_data, real_data.n_features, INIT_SEEDS_REAL, train_cfg)
    results["concrete"] = real_res

    out_real = paths.results_logs / "E3_capacity_concrete.json"
    with open(out_real, "w", encoding="utf-8") as f:
        json.dump(to_serializable({"experiment": "E3_capacity_concrete", **real_res}), f, indent=2)
    print(f"  Saved -> {out_real}")

    return results


if __name__ == "__main__":
    run_experiment()
