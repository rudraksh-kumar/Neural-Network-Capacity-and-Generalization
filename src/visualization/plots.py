"""Comprehensive visualization module for neural network capacity experiments.

Every public function produces one or more publication-quality PNG figures saved
to ``results/figures/`` and returns the ``matplotlib.figure.Figure`` handle(s).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config.experiment_config import ProjectPaths, SyntheticDataConfig

# ---------------------------------------------------------------------------
# Colour palette & style helpers
# ---------------------------------------------------------------------------

PALETTE = {
    "train": "#1f77b4",
    "val": "#ff7f0e",
    "test": "#2ca02c",
    "linear": "#d62728",
    "nn": "#9467bd",
    "noise_band": "#c7c7c7",
    "ground_truth": "#17becf",
    "emphasis": "#e377c2",
    "normal": "#1f77b4",
    "randomized": "#d62728",
    "protocol_a": "#1f77b4",
    "protocol_b": "#ff7f0e",
}

_MODEL_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def _colour_cycle(idx: int) -> str:
    return _MODEL_COLOURS[idx % len(_MODEL_COLOURS)]


def set_plot_style() -> None:
    """Configure matplotlib rcParams for publication-quality output."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "figure.figsize": (7, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _ensure_dir(save_dir: str | Path) -> Path:
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 1. Synthetic ground-truth landscape
# ---------------------------------------------------------------------------

def plot_synthetic_ground_truth(
    dataset: Any,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """fig01: Ground-truth curve, noisy training data, ±σ noise band, splits.

    Parameters
    ----------
    dataset : DatasetSplit
        Synthetic dataset built by ``build_synthetic_dataset``.
    save_dir : str | Path | None
        Directory for saving the figure. Defaults to ``results/figures/``.
    """
    set_plot_style()
    save_dir = _ensure_dir(save_dir or ProjectPaths().results_figures)
    cfg = SyntheticDataConfig()

    x_dense = np.linspace(cfg.x_min, cfg.x_max, 500).reshape(-1, 1)
    from src.data.synthetic_data import ground_truth_function
    f_dense = ground_truth_function(x_dense)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Noise band
    ax.fill_between(
        x_dense.ravel(),
        (f_dense - cfg.noise_std).ravel(),
        (f_dense + cfg.noise_std).ravel(),
        color=PALETTE["noise_band"], alpha=0.5, label=f"±σ noise band (σ={cfg.noise_std})",
    )
    # Ground truth
    ax.plot(x_dense.ravel(), f_dense.ravel(), color=PALETTE["ground_truth"],
            linewidth=2, label="f*(x) = 2 sin(2x) + 0.5x²")
    # Training points
    ax.scatter(dataset.X_train.ravel(), dataset.y_train.ravel(), s=18,
               color=PALETTE["train"], alpha=0.7, zorder=5,
               label=f"Train (N={len(dataset.X_train)})")
    # Val points (subsample for clarity)
    n_show_val = min(200, len(dataset.X_val))
    ax.scatter(dataset.X_val[:n_show_val].ravel(),
               dataset.y_val[:n_show_val].ravel(),
               s=8, color=PALETTE["val"], alpha=0.3, marker="x",
               label=f"Val (N={len(dataset.X_val)})")
    # Test points (subsample)
    n_show_test = min(200, len(dataset.X_test))
    ax.scatter(dataset.X_test[:n_show_test].ravel(),
               dataset.y_test[:n_show_test].ravel(),
               s=8, color=PALETTE["test"], alpha=0.2, marker="^",
               label=f"Test (N={len(dataset.X_test)})")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Synthetic Regression: Ground Truth and Data Splits")
    ax.legend(loc="upper left", fontsize=8)

    fig.savefig(save_dir / "fig01_synthetic_ground_truth.png")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 2. Gradient verification
# ---------------------------------------------------------------------------

def plot_gradient_verification(
    grad_results: dict[str, Any],
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """fig02: Bar chart of log10 relative errors per parameter for tanh/relu.

    Parameters
    ----------
    grad_results : dict
        Keys ``"tanh"`` and ``"relu"``, each mapping to a dict with
        ``"param_names"`` (list[str]) and ``"relative_errors"`` (list[float]).
    """
    set_plot_style()
    save_dir = _ensure_dir(save_dir or ProjectPaths().results_figures)

    activations = [k for k in ("tanh", "relu") if k in grad_results]
    n_act = len(activations)

    fig, axes = plt.subplots(1, n_act, figsize=(6 * n_act, 5), squeeze=False)

    for i, act in enumerate(activations):
        ax = axes[0, i]
        info = grad_results[act]
        names = info.get("param_names", [f"p{j}" for j in range(len(info["relative_errors"]))])
        errors = np.array(info["relative_errors"], dtype=np.float64)
        digits = -np.log10(np.clip(errors, 1e-16, None))
        thresh = 1e-7 if act == "tanh" else 1e-5
        thresh_digits = -np.log10(thresh)

        colours = ["#2ca02c" if e < thresh else "#d62728" for e in errors]
        bars = ax.bar(range(len(digits)), digits, color=colours, edgecolor="k", linewidth=0.6, alpha=0.85)
        for b, e in zip(bars, errors):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.18, f"{e:.1e}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(thresh_digits, color="#d62728", linestyle="--", linewidth=1.2, label=f"Tolerance ({thresh:.0e}, {thresh_digits:.0f} digits)")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim(0, max(digits.max() + 1.8, thresh_digits + 2))
        ax.set_ylabel("Precision: −log₁₀(relative error)")
        ax.set_title(f"Gradient Check — {act} (λ = 0)")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Numerical Gradient Verification", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_dir / "fig02_gradient_verification.png")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 3. Linear vs NN comparison
# ---------------------------------------------------------------------------

def plot_linear_vs_nn(
    baseline_results: dict[str, Any],
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """fig03: Synthetic curve fits overlay + concrete bar chart comparison.

    Parameters
    ----------
    baseline_results : dict
        Expected structure::

            {
              "synthetic": {
                "X_train": [...], "y_train": [...],
                "X_plot": [...], "f_true": [...],
                "linear_pred": [...], "nn_pred": [...],
                "linear_metrics": {"mse": ..., "r2": ...},
                "nn_metrics": {"mse": ..., "r2": ...},
              },
              "concrete": {
                "linear_metrics": {"mse": ..., "r2": ...},
                "nn_metrics": {"mse": ..., "r2": ...},
              },
            }
    """
    set_plot_style()
    save_dir = _ensure_dir(save_dir or ProjectPaths().results_figures)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left panel: Synthetic curve fits ---
    ax = axes[0]
    syn = baseline_results.get("synthetic", {})
    if syn:
        X_plot = np.asarray(syn.get("X_plot", []))
        f_true = np.asarray(syn.get("f_true", []))
        lin_pred = np.asarray(syn.get("linear_pred", []))
        nn_pred = np.asarray(syn.get("nn_pred", []))
        X_train = np.asarray(syn.get("X_train", []))
        y_train = np.asarray(syn.get("y_train", []))

        if X_train.size:
            ax.scatter(X_train.ravel(), y_train.ravel(), s=12, color=PALETTE["train"],
                       alpha=0.5, label="Train data")
        if X_plot.size and f_true.size:
            sort_idx = np.argsort(X_plot.ravel())
            ax.plot(X_plot.ravel()[sort_idx], f_true.ravel()[sort_idx],
                    color=PALETTE["ground_truth"], linewidth=2, label="f*(x)")
        if X_plot.size and lin_pred.size:
            sort_idx = np.argsort(X_plot.ravel())
            ax.plot(X_plot.ravel()[sort_idx], lin_pred.ravel()[sort_idx],
                    color=PALETTE["linear"], linewidth=1.5, linestyle="--", label="Linear")
        if X_plot.size and nn_pred.size:
            sort_idx = np.argsort(X_plot.ravel())
            ax.plot(X_plot.ravel()[sort_idx], nn_pred.ravel()[sort_idx],
                    color=PALETTE["nn"], linewidth=1.5, label="MLP")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Synthetic: Curve Fits")
    ax.legend(fontsize=8)

    # --- Right panel: Concrete bar chart ---
    ax2 = axes[1]
    datasets_for_bars = []
    for ds_key in ("synthetic", "concrete"):
        ds = baseline_results.get(ds_key, {})
        if ds:
            lm = ds.get("linear_metrics", {})
            nm = ds.get("nn_metrics", {})
            datasets_for_bars.append((ds_key, lm, nm))

    if datasets_for_bars:
        x_positions = np.arange(len(datasets_for_bars))
        bar_w = 0.35
        lin_mse = [d[1].get("mse", 0) for d in datasets_for_bars]
        nn_mse = [d[2].get("mse", 0) for d in datasets_for_bars]
        ax2.bar(x_positions - bar_w / 2, lin_mse, bar_w, color=PALETTE["linear"], label="Linear")
        ax2.bar(x_positions + bar_w / 2, nn_mse, bar_w, color=PALETTE["nn"], label="MLP")
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels([d[0].title() for d in datasets_for_bars])
        ax2.set_ylabel("Val MSE")
        ax2.set_title("Baseline Comparison")
        ax2.legend()

    fig.suptitle("E2: Linear Baseline vs Neural Network", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_dir / "fig03_linear_vs_nn.png")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 4. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    histories_dict: dict[str, dict[str, list[float]]],
    save_dir: str | Path | None = None,
    title_suffix: str = "",
) -> plt.Figure:
    """fig04: Epoch vs train/val MSE for multiple models.

    Parameters
    ----------
    histories_dict : dict[str, dict[str, list[float]]]
        ``{model_name: {"train_loss": [...], "val_loss": [...]}}``.
    title_suffix : str
        Extra string appended to the figure title.
    """
    set_plot_style()
    save_dir = _ensure_dir(save_dir or ProjectPaths().results_figures)

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (name, hist) in enumerate(histories_dict.items()):
        c = _colour_cycle(idx)
        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], color=c, linewidth=1.2,
                label=f"{name} train")
        ax.plot(epochs, hist["val_loss"], color=c, linewidth=1.2,
                linestyle="--", label=f"{name} val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (units²)" if "synthetic" in title_suffix.lower() else "Standardized MSE (training scale)")
    ax.set_yscale("log")
    ax.set_title(f"Training Curves{(' — ' + title_suffix) if title_suffix else ''}")
    ax.legend(fontsize=7, ncol=2)

    fname = "fig04_training_curves"
    if title_suffix:
        fname += "_" + title_suffix.lower().replace(" ", "_")
    fig.savefig(save_dir / f"{fname}.png")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 5. Width vs performance
# ---------------------------------------------------------------------------

def plot_width_vs_performance(
    width_results: dict[str, Any],
    save_dir: str | Path | None = None,
    dataset_name: str = "synthetic",
) -> plt.Figure:
    """fig05: Width H vs train/val MSE (μ±σ) with Bayes-optimal floor.

    Parameters
    ----------
    width_results : dict
        Expected keys: ``"widths"`` (list[int]),
        ``"train_mse_mean"``, ``"train_mse_std"``,
        ``"val_mse_mean"``, ``"val_mse_std"`` (each list[float]),
        and optionally ``"irreducible_mse"`` (float).
    """
    set_plot_style()
    save_dir = _ensure_dir(save_dir or ProjectPaths().results_figures)

    widths = np.array(width_results["widths"])
    tr_mean = np.array(width_results["train_mse_mean"])
    tr_std = np.array(width_results["train_mse_std"])
    val_mean = np.array(width_results["val_mse_mean"])
    val_std = np.array(width_results["val_mse_std"])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(widths, tr_mean,
                yerr=[np.minimum(tr_std, tr_mean * 0.7), tr_std],
                fmt="o-", color=PALETTE["train"], capsize=3, label="Train MSE")
    ax.errorbar(widths, val_mean,
                yerr=[np.minimum(val_std, val_mean * 0.7), val_std],
                fmt="s--", color=PALETTE["val"], capsize=3, label="Val MSE")

    irr = width_results.get("irreducible_mse")
    if irr is not None:
        ax.axhline(irr, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Bayes floor (σ²={irr:.4f})")

    unit_sq = "units²" if dataset_name.lower() == "synthetic" else "MPa²"
    ax.set_xlabel("Hidden Width H")
    ax.set_ylabel(f"MSE ({unit_sq})")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_title(f"E3: Width vs Performance — {dataset_name.title()}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_dir / f"fig05_width_vs_performance_{dataset_name}.png")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 6. Sample-size curves
# ---------------------------------------------------------------------------

def plot_sample_size_curves(
    size_results: dict[str, Any],
    save_dir: str | Path | None = None,
    dataset_name: str = "synthetic",
) -> plt.Figure:
    """fig06: N vs train/val MSE (μ±σ).

    Parameters
    ----------
    size_results : dict
        Keys: ``"sample_sizes"``, ``"train_mse_mean"``, ``"train_mse_std"``,
        ``"val_mse_mean"``, ``"val_mse_std"``.
    """
    set_plot_style()
    save_dir = _ensure_dir(save_dir or ProjectPaths().results_figures)

    sizes = np.array(size_results["sample_sizes"])
    tr_mean = np.array(size_results["train_mse_mean"])
    tr_std = np.array(size_results["train_mse_std"])
    val_mean = np.array(size_results["val_mse_mean"])
    val_std = np.array(size_results["val_mse_std"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sizes, tr_mean,
                yerr=[np.minimum(tr_std, tr_mean * 0.7), tr_std],
                fmt="o-", color=PALETTE["train"], capsize=3, label="Train MSE")
    ax.errorbar(sizes, val_mean,
                yerr=[np.minimum(val_std, val_mean * 0.7), val_std],
                fmt="s--", color=PALETTE["val"], capsize=3, label="Val MSE")

    if "linear_val_mse_mean" in size_results:
        ax.plot(sizes, size_results["linear_val_mse_mean"], "^:",
                color=PALETTE["linear"], linewidth=1.4, label="Linear OLS Val MSE")

    irr = size_results.get("irreducible_mse")
    if irr is not None:
        ax.axhline(irr, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Bayes floor (σ²={irr:.4f})")

    unit_sq = "units²" if dataset_name.lower() == "synthetic" else "MPa²"
    ax.set_xlabel("Training Sample Size N")
    ax.set_ylabel(f"MSE ({unit_sq})")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_title(f"E4: Sample Size vs Performance — {dataset_name.title()}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_dir / f"fig06_sample_size_{dataset_name}.png")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 7. Generalization gap
# ---------------------------------------------------------------------------

def plot_generalization_gap(
    width_results: dict[str, Any],
    save_dir: str | Path | None = None,
    dataset_name: str = "synthetic",
) -> plt.Figure:
    """fig07: Generalization gap and P/N ratio vs hidden width.

    Parameters
    ----------
    width_results : dict
        Keys: ``"widths"``, ``"val_mse_mean"``, ``"train_mse_mean"``,
        ``"n_train"`` (int), and optionally ``"param_counts"`` (list[int]).
    """
    set_plot_style()
    save_dir = _ensure_dir(save_dir or ProjectPaths().results_figures)

    widths = np.array(width_results["widths"])
    gap = np.array(width_results["val_mse_mean"]) - np.array(width_results["train_mse_mean"])
    n_train = width_results.get("n_train", 100)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    unit_sq = "units²" if dataset_name.lower() == "synthetic" else "MPa²"
    colour_gap = PALETTE["val"]
    ax1.plot(widths, gap, "o-", color=colour_gap, label="Gen. gap (val − train MSE)")
    ax1.set_xlabel("Hidden Width H")
    ax1.set_ylabel(f"Generalization Gap (MSE, {unit_sq})", color=colour_gap)
    ax1.tick_params(axis="y", labelcolor=colour_gap)
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # P/N ratio on secondary axis
    ax2 = ax1.twinx()
    if "param_counts" in width_results:
        params = np.array(width_results["param_counts"])
    else:
        # 1-hidden-layer MLP: P = (d+1)*H + (H+1) for input_dim=d
        input_dim = width_results.get("input_dim", 1)
        params = (input_dim + 1) * widths + (widths + 1)
    pn_ratio = params / n_train

    colour_pn = PALETTE["emphasis"]
    ax2.plot(widths, pn_ratio, "^--", color=colour_pn, label="P / N")
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax2.set_ylabel("P / N ratio", color=colour_pn)
    ax2.tick_params(axis="y", labelcolor=colour_pn)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    ax1.set_title(f"E3: Generalization Gap — {dataset_name.title()}")
    fig.tight_layout()
    fig.savefig(save_dir / f"fig07_generalization_gap_{dataset_name}.png")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 8. Weight-decay comparison
# ---------------------------------------------------------------------------

def plot_weight_decay_comparison(
    reg_results: dict[str, Any],
    save_dir: str | Path | None = None,
    dataset_name: str = "synthetic",
) -> plt.Figure:
    """fig08: λ sweep effects on train / val MSE.

    Parameters
    ----------
    reg_results : dict
        Keys: ``"lambdas"`` (list[float]),
        ``"train_mse_mean"``, ``"train_mse_std"``,
        ``"val_mse_mean"``, ``"val_mse_std"`` (each list[float]).
        Optionally ``"adamw_train_mse"``, ``"adamw_val_mse"`` for AdamW overlay.
    """
    set_plot_style()
    save_dir = _ensure_dir(save_dir or ProjectPaths().results_figures)

    lambdas = np.array(reg_results["lambdas"])
    tr_mean = np.array(reg_results["train_mse_mean"])
    tr_std = np.array(reg_results["train_mse_std"])
    val_mean = np.array(reg_results["val_mse_mean"])
    val_std = np.array(reg_results["val_mse_std"])

    fig, ax = plt.subplots(figsize=(8, 5))

    # Shift zero to small positive for log scale
    x_labels = [f"{l:.0e}" if l > 0 else "0" for l in lambdas]
    x_pos = np.arange(len(lambdas))

    ax.errorbar(x_pos, tr_mean, yerr=tr_std, fmt="o-",
                color=PALETTE["train"], capsize=3, label="Train MSE (Adam+L2)")
    ax.errorbar(x_pos, val_mean, yerr=val_std, fmt="s--",
                color=PALETTE["val"], capsize=3, label="Val MSE (Adam+L2)")

    # Optional AdamW comparison
    if "adamw_val_mse" in reg_results:
        aw_tr = reg_results.get("adamw_train_mse", [None] * len(lambdas))
        aw_val = reg_results["adamw_val_mse"]
        ax.plot(x_pos, aw_val, "D-.", color=PALETTE["emphasis"],
                label="Val MSE (AdamW)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30)
    ax.set_xlabel("Weight Decay λ")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title(f"E5: Weight Decay Sweep — {dataset_name.title()}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_dir / f"fig08_weight_decay_{dataset_name}.png")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 9. Early stopping
# ---------------------------------------------------------------------------

def plot_early_stopping(
    histories: dict[str, Any],
    save_dir: str | Path | None = None,
    dataset_name: str = "synthetic",
) -> plt.Figure:
    """fig09: Loss curves with stopping-epoch markers.

    Parameters
    ----------
    histories : dict
        ``{label: {"train_loss": [...], "val_loss": [...],
                   "stopped_epoch": int | None}}``.
    """
    set_plot_style()
    save_dir = _ensure_dir(save_dir or ProjectPaths().results_figures)

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, h) in enumerate(histories.items()):
        c = _colour_cycle(idx)
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], color=c, linewidth=1.2,
                label=f"{label} train")
        ax.plot(epochs, h["val_loss"], color=c, linestyle="--",
                linewidth=1.2, label=f"{label} val")

        stopped = h.get("stopped_epoch")
        if stopped is not None:
            ax.axvline(stopped, color=c, linestyle=":", linewidth=1,
                       alpha=0.7)
            ax.annotate(f"stop@{stopped}", xy=(stopped, h["val_loss"][min(stopped - 1, len(h['val_loss']) - 1)]),
                        fontsize=7, color=c,
                        xytext=(5, 10), textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color=c, lw=0.8))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (units²)" if "synthetic" in title_suffix.lower() else "Standardized MSE (training scale)")
    ax.set_yscale("log")
    ax.set_title(f"E5: Early Stopping — {dataset_name.title()}")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(save_dir / f"fig09_early_stopping_{dataset_name}.png")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 10. Randomized labels
# ---------------------------------------------------------------------------

def plot_randomized_labels(
    random_results: dict[str, Any],
    save_dir: str | Path | None = None,
    dataset_name: str = "synthetic",
) -> plt.Figure:
    """fig10: Normal vs randomized targets across widths.

    Parameters
    ----------
    random_results : dict
        Keys: ``"widths"`` (list[int]),
        ``"normal_train_mse"``, ``"normal_val_mse"``,
        ``"random_train_mse"``, ``"random_val_mse"`` (each list[float]).
    """
    set_plot_style()
    save_dir = _ensure_dir(save_dir or ProjectPaths().results_figures)

    widths = np.array(random_results["widths"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Left: Train MSE
    ax = axes[0]
    ax.plot(widths, random_results["normal_train_mse"], "o-",
            color=PALETTE["normal"], label="Normal")
    ax.plot(widths, random_results["random_train_mse"], "s--",
            color=PALETTE["randomized"], label="Randomized")
    ax.set_xlabel("Hidden Width H")
    ax.set_ylabel("Train MSE")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_title("Training Error")
    ax.legend()

    # Right: Val MSE
    ax = axes[1]
    ax.plot(widths, random_results["normal_val_mse"], "o-",
            color=PALETTE["normal"], label="Normal")
    ax.plot(widths, random_results["random_val_mse"], "s--",
            color=PALETTE["randomized"], label="Randomized")
    ax.set_xlabel("Hidden Width H")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_title("Validation Error (True Targets)")
    ax.legend()

    fig.suptitle(f"E6: Randomized Labels Experiment — {dataset_name.title()}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_dir / f"fig10_randomized_labels_{dataset_name}.png")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 11. Computational budget
# ---------------------------------------------------------------------------

def plot_computational_budget(
    budget_results: dict[str, Any],
    save_dir: str | Path | None = None,
    dataset_name: str = "synthetic",
) -> plt.Figure:
    """fig11: Protocol A (fixed step budget) vs Protocol B (convergence) comparisons.

    Parameters
    ----------
    budget_results : dict
        Keys: ``"widths"`` (list[int]),
        ``"protocol_a_val_mse"`` (list[float]),
        ``"protocol_b_val_mse"`` (list[float]).
        Optionally: ``"protocol_a_train_mse"``, ``"protocol_b_train_mse"``.
    """
    set_plot_style()
    save_dir = _ensure_dir(save_dir or ProjectPaths().results_figures)

    widths = np.array(budget_results["widths"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Val MSE comparison
    ax = axes[0]
    ax.plot(widths, budget_results["protocol_a_val_mse"], "o-",
            color=PALETTE["protocol_a"], label="Protocol A (fixed budget)")
    ax.plot(widths, budget_results["protocol_b_val_mse"], "s--",
            color=PALETTE["protocol_b"], label="Protocol B (convergence)")
    ax.set_xlabel("Hidden Width H")
    ax.set_ylabel("Val MSE")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_title("Validation MSE: A vs B")
    ax.legend()

    # Right: Train MSE comparison
    ax = axes[1]
    tr_a = budget_results.get("protocol_a_train_mse")
    tr_b = budget_results.get("protocol_b_train_mse")
    if tr_a is not None and tr_b is not None:
        ax.plot(widths, tr_a, "o-", color=PALETTE["protocol_a"],
                label="Protocol A train")
        ax.plot(widths, tr_b, "s--", color=PALETTE["protocol_b"],
                label="Protocol B train")
        ax.set_xlabel("Hidden Width H")
        ax.set_ylabel("Train MSE")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_yscale("log")
        ax.set_title("Training MSE: A vs B")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No train MSE data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Training MSE (no data)")

    fig.suptitle(f"E7: Computational Budget — {dataset_name.title()}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_dir / f"fig11_computational_budget_{dataset_name}.png")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 12. Diagnostic interventions (multi-panel)
# ---------------------------------------------------------------------------

def plot_diagnostics(
    diag_results: dict[str, Any],
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """fig12: Multi-panel diagnostic interventions.

    Parameters
    ----------
    diag_results : dict
        Expected sub-keys: ``"lr_sweep"``, ``"optimizer_sweep"``,
        ``"width_under_budget"``.  Each sub-dict has keys appropriate for a
        small bar chart or line plot.
    """
    set_plot_style()
    save_dir = _ensure_dir(save_dir or ProjectPaths().results_figures)

    n_panels = sum(1 for k in ("lr_sweep", "optimizer_sweep", "width_under_budget")
                   if k in diag_results)
    n_panels = max(n_panels, 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), squeeze=False)
    panel_idx = 0

    # --- LR sweep ---
    if "lr_sweep" in diag_results:
        ax = axes[0, panel_idx]
        lr_info = diag_results["lr_sweep"]
        lrs = lr_info.get("learning_rates", [])
        val_mse = lr_info.get("val_mse", [])
        train_mse = lr_info.get("train_mse", [])
        x_pos = np.arange(len(lrs))
        labels = [f"{lr:.0e}" for lr in lrs]
        if train_mse:
            ax.bar(x_pos - 0.2, train_mse, 0.35, color=PALETTE["train"], label="Train")
        if val_mse:
            ax.bar(x_pos + 0.2, val_mse, 0.35, color=PALETTE["val"], label="Val")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.set_title("LR Sweep")
        ax.legend(fontsize=7)
        panel_idx += 1

    # --- Optimizer sweep ---
    if "optimizer_sweep" in diag_results:
        ax = axes[0, panel_idx]
        opt_info = diag_results["optimizer_sweep"]
        optimizers = opt_info.get("optimizers", [])
        val_mse = opt_info.get("val_mse", [])
        train_mse = opt_info.get("train_mse", [])
        x_pos = np.arange(len(optimizers))
        if train_mse:
            ax.bar(x_pos - 0.2, train_mse, 0.35, color=PALETTE["train"], label="Train")
        if val_mse:
            ax.bar(x_pos + 0.2, val_mse, 0.35, color=PALETTE["val"], label="Val")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(optimizers)
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.set_title("Optimizer Comparison")
        ax.legend(fontsize=7)
        panel_idx += 1

    # --- Width under fixed budget ---
    if "width_under_budget" in diag_results:
        ax = axes[0, panel_idx]
        wb_info = diag_results["width_under_budget"]
        widths = wb_info.get("widths", [])
        val_mse = wb_info.get("val_mse", [])
        train_mse = wb_info.get("train_mse", [])
        if widths:
            if train_mse:
                ax.plot(widths, train_mse, "o-", color=PALETTE["train"], label="Train")
            if val_mse:
                ax.plot(widths, val_mse, "s--", color=PALETTE["val"], label="Val")
            ax.set_xlabel("Hidden Width H")
            ax.set_ylabel("MSE")
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_yscale("log")
            ax.set_title("Width under Fixed Budget")
            ax.legend(fontsize=7)
        panel_idx += 1

    fig.suptitle("E8: Diagnostic Interventions", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_dir / "fig12_diagnostics.png")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 13. Real-data predictions
# ---------------------------------------------------------------------------

def plot_real_predictions(
    y_true: np.ndarray,
    y_pred_dict: dict[str, np.ndarray],
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """fig13: Predicted vs actual scatter + residuals for concrete dataset.

    Parameters
    ----------
    y_true : np.ndarray
        True target values (MPa).
    y_pred_dict : dict[str, np.ndarray]
        ``{model_name: predictions_array}``.
    """
    set_plot_style()
    save_dir = _ensure_dir(save_dir or ProjectPaths().results_figures)

    n_models = len(y_pred_dict)
    fig, axes = plt.subplots(2, max(n_models, 1), figsize=(5 * max(n_models, 1), 9),
                             squeeze=False)
    yt = np.asarray(y_true).ravel()

    for idx, (name, y_pred) in enumerate(y_pred_dict.items()):
        yp = np.asarray(y_pred).ravel()
        c = _colour_cycle(idx)

        # Top: pred vs actual
        ax = axes[0, idx]
        ax.scatter(yt, yp, s=10, alpha=0.6, color=c)
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
        ax.set_xlabel("Actual (MPa)")
        ax.set_ylabel("Predicted (MPa)")
        ax.set_title(f"{name}")
        ax.legend(fontsize=7)

        # Bottom: residuals
        ax = axes[1, idx]
        residuals = yp - yt
        ax.scatter(yt, residuals, s=10, alpha=0.6, color=c)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_xlabel("Actual (MPa)")
        ax.set_ylabel("Residual (MPa)")
        ax.set_title(f"{name} — Residuals")

    fig.suptitle("Concrete: Predicted vs Actual", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_dir / "fig13_real_predictions.png")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 14. Final comparison bar chart
# ---------------------------------------------------------------------------

def plot_final_comparison(
    final_results: dict[str, Any],
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """fig14: Final test metrics comparison bar chart.

    Parameters
    ----------
    final_results : dict
        ``{model_name: {"test_mse": float, "test_r2": float}}``.
    """
    set_plot_style()
    save_dir = _ensure_dir(save_dir or ProjectPaths().results_figures)

    names = list(final_results.keys())
    mse_vals = [final_results[n].get("test_mse", 0) for n in names]
    r2_vals = [final_results[n].get("test_r2", 0) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_pos = np.arange(len(names))
    colours = [_colour_cycle(i) for i in range(len(names))]

    # MSE
    ax = axes[0]
    ax.bar(x_pos, mse_vals, color=colours, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test MSE")
    ax.set_title("Test MSE Comparison")

    # R²
    ax = axes[1]
    ax.bar(x_pos, r2_vals, color=colours, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test R²")
    ax.set_title("Test R² Comparison")
    ax.set_ylim(0, 1.05)

    fig.suptitle("Final Model Comparison on Held-Out Test Set", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_dir / "fig14_final_comparison.png")
    plt.close(fig)
    return fig
