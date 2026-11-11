"""E6: Randomized-label memorization vs generalization experiment.

Independently permutes training targets y_train while preserving X_train and
keeping validation and test targets completely unpermuted.

Includes a 1D spectral-bias diagnostic comparing standard Xavier (b1=0) vs
distributed high-frequency initialization on the 1D synthetic dataset.
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
)
from src.data.real_data import load_concrete_dataset
from src.data.synthetic_data import build_synthetic_dataset
from src.evaluation.metrics import (
    aggregate_seed_metrics,
    evaluate_model_physical,
    to_serializable,
)
from src.models.neural_network_numpy import NeuralNetworkNumPy
from src.training.trainer import train_model


WIDTHS_E6 = [4, 16, 64, 128, 256]
SIZES_E6 = [25, 50, 100, 200]
MEM_THRESHOLD_SCALED_MSE = 0.05


def _first_epoch_below(curve: list[float], threshold: float) -> int | None:
    for idx, val in enumerate(curve):
        if val < threshold:
            return idx + 1
    return None


def _run_grid(dataset_name: str, seeds: list[int], train_cfg: TrainConfig) -> list[dict]:
    syn_cfg = SyntheticDataConfig()
    real_cfg = RealDataConfig()
    all_results: list[dict] = []

    for randomize in (False, True):
        label = "random" if randomize else "normal"
        for N in SIZES_E6:
            for H in WIDTHS_E6:
                seed_records: list[dict] = []
                for seed in seeds:
                    if dataset_name == "synthetic":
                        data = build_synthetic_dataset(
                            cfg=syn_cfg, n_train=N, randomize_train_labels=randomize
                        )
                        input_dim = 1
                    else:
                        data = load_concrete_dataset(
                            cfg=real_cfg, n_train_subsample=N, randomize_train_labels=randomize
                        )
                        input_dim = data.n_features

                    mlp = NeuralNetworkNumPy(
                        input_dim=input_dim,
                        hidden_width=H,
                        activation="tanh",
                        seed=seed,
                    )
                    hist = train_model(
                        mlp, data.X_train, data.y_train, data.X_val, data.y_val, cfg=train_cfg
                    )
                    # Train error measured against the assigned (normal or randomized) targets
                    tr_m = evaluate_model_physical(mlp, data.X_train, data.y_train, data)
                    # Val and Test always measured against original unpermuted targets
                    val_m = evaluate_model_physical(mlp, data.X_val, data.y_val, data)
                    test_m = evaluate_model_physical(mlp, data.X_test, data.y_test, data)

                    ep_fit = _first_epoch_below(hist.train_mse, MEM_THRESHOLD_SCALED_MSE)
                    seed_records.append({
                        "seed": seed,
                        "train_metrics": tr_m,
                        "val_metrics": val_m,
                        "test_metrics": test_m,
                        "epochs_to_fit": ep_fit,
                        "memorized": ep_fit is not None,
                    })

                tr_agg = aggregate_seed_metrics([r["train_metrics"] for r in seed_records])
                val_agg = aggregate_seed_metrics([r["val_metrics"] for r in seed_records])
                test_agg = aggregate_seed_metrics([r["test_metrics"] for r in seed_records])
                fit_epochs = [r["epochs_to_fit"] for r in seed_records if r["epochs_to_fit"] is not None]
                mem_rate = float(np.mean([1.0 if r["memorized"] else 0.0 for r in seed_records]))

                all_results.append({
                    "labels": label,
                    "n_train": N,
                    "hidden_width": H,
                    "memorization_rate": mem_rate,
                    "mean_epochs_to_fit": float(np.mean(fit_epochs)) if fit_epochs else None,
                    "seed_runs": seed_records,
                    "train_aggregate": tr_agg,
                    "val_aggregate": val_agg,
                    "test_aggregate": test_agg,
                })
                if N in (25, 100):
                    print(
                        f"  [{dataset_name[:3]}] {label:<6s} N={N:>3d} H={H:>3d} | "
                        f"train_mse={tr_agg['mse_mean']:.4f} | val_mse={val_agg['mse_mean']:.4f} | "
                        f"test_mse={test_agg['mse_mean']:.4f} | mem_rate={mem_rate:.2f}"
                    )

    return all_results


def _run_1d_spectral_bias_check(seeds: list[int]) -> list[dict]:
    """Compare standard Xavier (b1=0) vs distributed high-freq init on 1D random labels."""
    syn_cfg = SyntheticDataConfig()
    cfg = TrainConfig(optimizer="adam", learning_rate=0.03, max_epochs=3500, eval_val_every=50)
    rows: list[dict] = []
    for N in (25, 50, 100):
        for mode in ("xavier_b0", "high_freq_spread_bias"):
            tr_ms, val_ms = [], []
            for seed in seeds:
                data = build_synthetic_dataset(cfg=syn_cfg, n_train=N, randomize_train_labels=True)
                mlp = NeuralNetworkNumPy(input_dim=1, hidden_width=256, activation="tanh", seed=seed)
                if mode == "high_freq_spread_bias":
                    rng = np.random.default_rng(seed)
                    mlp.W1 = rng.normal(0.0, 12.0, size=mlp.W1.shape)
                    mlp.b1 = rng.uniform(-30.0, 30.0, size=mlp.b1.shape)
                train_model(mlp, data.X_train, data.y_train, data.X_val, data.y_val, cfg=cfg)
                tr_ms.append(evaluate_model_physical(mlp, data.X_train, data.y_train, data))
                val_ms.append(evaluate_model_physical(mlp, data.X_val, data.y_val, data))
            rows.append({
                "n_train": N,
                "init_mode": mode,
                "train_aggregate": aggregate_seed_metrics(tr_ms),
                "val_aggregate": aggregate_seed_metrics(val_ms),
            })
    return rows


def run_experiment() -> dict:
    paths = ProjectPaths()
    paths.ensure_dirs()

    train_cfg = TrainConfig(
        optimizer="adam",
        learning_rate=0.01,
        batch_size=None,
        max_epochs=2500,
        eval_val_every=25,
    )

    print("[E6] Synthetic randomized-label sweep ...")
    syn_res = _run_grid("synthetic", INIT_SEEDS_SYN[:5], train_cfg)
    syn_spectral = _run_1d_spectral_bias_check(INIT_SEEDS_SYN[:3])
    out_syn = paths.results_logs / "E6_randomized_synthetic.json"
    with open(out_syn, "w", encoding="utf-8") as f:
        json.dump(
            to_serializable({
                "experiment": "E6_randomized_synthetic",
                "widths": WIDTHS_E6,
                "sizes": SIZES_E6,
                "results": syn_res,
                "spectral_bias_diagnostic": syn_spectral,
            }),
            f,
            indent=2,
        )
    print(f"  Saved -> {out_syn}")

    print("[E6] Concrete randomized-label sweep ...")
    real_res = _run_grid("concrete", INIT_SEEDS_REAL, train_cfg)
    out_real = paths.results_logs / "E6_randomized_concrete.json"
    with open(out_real, "w", encoding="utf-8") as f:
        json.dump(
            to_serializable({
                "experiment": "E6_randomized_concrete",
                "widths": WIDTHS_E6,
                "sizes": SIZES_E6,
                "results": real_res,
            }),
            f,
            indent=2,
        )
    print(f"  Saved -> {out_real}")

    return {"experiment": "E6_randomized_labels", "synthetic": syn_res, "concrete": real_res}


if __name__ == "__main__":
    run_experiment()
