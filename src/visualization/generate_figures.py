"""Generate all 14+ required publication-quality figures directly from experiment JSON logs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from config.experiment_config import ProjectPaths, SyntheticDataConfig
from src.data.synthetic_data import build_synthetic_dataset
from src.visualization.plots import (
    PALETTE,
    _colour_cycle,
    plot_generalization_gap,
    plot_gradient_verification,
    plot_real_predictions,
    plot_sample_size_curves,
    plot_synthetic_ground_truth,
    plot_training_curves,
    plot_width_vs_performance,
    set_plot_style,
)


def _load_json(logs_dir: Path, name: str) -> dict | None:
    p = logs_dir / name
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_all_figures(
    logs_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
) -> list[Path]:
    """Read saved experiment logs and emit all required PNG figures."""
    paths = ProjectPaths()
    paths.ensure_dirs()
    ldir = Path(logs_dir) if logs_dir else paths.results_logs
    fdir = Path(figures_dir) if figures_dir else paths.results_figures
    fdir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    saved: list[Path] = []

    # 1. fig01: Synthetic dataset & ground truth
    syn_data = build_synthetic_dataset(cfg=SyntheticDataConfig())
    plot_synthetic_ground_truth(syn_data, save_dir=fdir)
    saved.append(fdir / "fig01_synthetic_ground_truth.png")

    # 2. fig02: Gradient verification
    e1 = _load_json(ldir, "E1_gradient_check.json")
    if e1:
        grad_for_plot = {}
        for chk in e1["checks"]:
            if chk["l2_lambda"] == 0.0:
                act = chk["activation"]
                names = list(chk["parameters"].keys())
                rels = [chk["parameters"][k]["rel_err_fd"] for k in names]
                grad_for_plot[act] = {"param_names": names, "relative_errors": rels}
        if grad_for_plot:
            plot_gradient_verification(grad_for_plot, save_dir=fdir)
            saved.append(fdir / "fig02_gradient_verification.png")

    # 3. fig03 & fig13: Linear vs NN baseline + Concrete predictions vs actual
    e2 = _load_json(ldir, "E2_baseline.json")
    if e2:
        syn_p = e2["plot_data"].get("synthetic", {})
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        if syn_p:
            ax.scatter(syn_p["X_train"], syn_p["y_train"], s=14, alpha=0.55, color=PALETTE["train"], label="Train data")
            ax.plot(syn_p["X_plot"], syn_p["f_true"], color=PALETTE["ground_truth"], lw=2.0, label="f*(x) ground truth")
            ax.plot(syn_p["X_plot"], syn_p["linear_pred"], "--", color=PALETTE["linear"], lw=1.8, label="Linear OLS")
            ax.plot(syn_p["X_plot"], syn_p["nn_tanh_pred"], color=PALETTE["nn"], lw=1.8, label="MLP (H=32, tanh)")
            if "nn_relu_pred" in syn_p:
                ax.plot(syn_p["X_plot"], syn_p["nn_relu_pred"], "-.", color=PALETTE["test"], lw=1.4, label="MLP (H=32, ReLU)")
        ax.set_xlabel("x (units)")
        ax.set_ylabel("y (units)")
        ax.set_title("Synthetic: Linear vs 1-HL MLP Fits")
        ax.legend(fontsize=8)

        ax2 = axes[1]
        agg = e2["aggregates"]
        labels = ["Synthetic\n(units)", "Concrete\n(MPa)"]
        lin_r2 = [agg["synthetic_linear"]["test"]["r2_mean"], agg["concrete_linear"]["test"]["r2_mean"]]
        tanh_r2 = [agg["synthetic_mlp_tanh"]["test"]["r2_mean"], agg["concrete_mlp_tanh"]["test"]["r2_mean"]]
        relu_r2 = [agg["synthetic_mlp_relu"]["test"]["r2_mean"], agg["concrete_mlp_relu"]["test"]["r2_mean"]]
        xp = np.arange(len(labels))
        w = 0.25
        ax2.bar(xp - w, lin_r2, w, color=PALETTE["linear"], label="Linear OLS")
        ax2.bar(xp, tanh_r2, w, color=PALETTE["nn"], label="MLP (H=32, tanh)")
        ax2.bar(xp + w, relu_r2, w, color=PALETTE["test"], label="MLP (H=32, ReLU)")
        ax2.set_xticks(xp)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Test R² Score")
        ax2.set_ylim(0, 1.08)
        ax2.set_title("Baseline Generalization (Test R²)")
        ax2.legend(fontsize=8)
        fig.suptitle("E2: Linear Baseline vs One-Hidden-Layer Neural Network", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(fdir / "fig03_linear_vs_nn.png")
        plt.close(fig)
        saved.append(fdir / "fig03_linear_vs_nn.png")

        con_p = e2["plot_data"].get("concrete", {})
        if con_p:
            plot_real_predictions(
                np.asarray(con_p["y_test_mpa"]),
                {
                    "Linear OLS": np.asarray(con_p["linear_pred_mpa"]),
                    "MLP (H=32, tanh)": np.asarray(con_p["nn_tanh_pred_mpa"]),
                },
                save_dir=fdir,
            )
            saved.append(fdir / "fig13_real_predictions.png")

    # 4, 5, 7. fig04, fig05, fig07 from E3 Capacity Sweep
    for dname in ("synthetic", "concrete"):
        e3 = _load_json(ldir, f"E3_capacity_{dname}.json")
        if not e3:
            continue
        rep_curves = e3.get("representative_curves", {})
        if rep_curves:
            histories = {
                k: {"train_loss": v["train_mse"], "val_loss": v["val_mse"]}
                for k, v in rep_curves.items()
            }
            plot_training_curves(histories, save_dir=fdir, title_suffix=dname)
            saved.append(fdir / f"fig04_training_curves_{dname}.png")

        wr = e3["width_results"]
        w_dict = {
            "widths": [r["hidden_width"] for r in wr],
            "param_counts": [r["n_params"] for r in wr],
            "n_train": e3["n_train"],
            "input_dim": e3["input_dim"],
            "train_mse_mean": [r["train_aggregate"]["mse_mean"] for r in wr],
            "train_mse_std": [r["train_aggregate"]["mse_std"] for r in wr],
            "val_mse_mean": [r["val_aggregate"]["mse_mean"] for r in wr],
            "val_mse_std": [r["val_aggregate"]["mse_std"] for r in wr],
            "irreducible_mse": e3.get("irreducible_mse"),
        }
        plot_width_vs_performance(w_dict, save_dir=fdir, dataset_name=dname)
        saved.append(fdir / f"fig05_width_vs_performance_{dname}.png")

        plot_generalization_gap(w_dict, save_dir=fdir, dataset_name=dname)
        saved.append(fdir / f"fig07_generalization_gap_{dname}.png")

    # 6. fig06 from E4 Sample-Size Sweep
    for dname in ("synthetic", "concrete"):
        e4 = _load_json(ldir, f"E4_sample_size_{dname}.json")
        if not e4:
            continue
        sr = e4["size_results"]
        s_dict = {
            "sample_sizes": [r["n_train"] for r in sr],
            "train_mse_mean": [r["train_aggregate"]["mse_mean"] for r in sr],
            "train_mse_std": [r["train_aggregate"]["mse_std"] for r in sr],
            "val_mse_mean": [r["val_aggregate"]["mse_mean"] for r in sr],
            "val_mse_std": [r["val_aggregate"]["mse_std"] for r in sr],
            "linear_val_mse_mean": [r["linear_val_aggregate"]["mse_mean"] for r in sr],
            "irreducible_mse": e4.get("irreducible_mse"),
        }
        plot_sample_size_curves(s_dict, save_dir=fdir, dataset_name=dname)
        saved.append(fdir / f"fig06_sample_size_{dname}.png")

    # 8 & 9. fig08 & fig09 from E5 Regularization
    for dname in ("synthetic", "concrete"):
        e5 = _load_json(ldir, f"E5_regularization_{dname}.json")
        if not e5:
            continue
        entries = e5["reg_results"]
        unreg = next(r for r in entries if r["label"] == "unreg")
        l2_entries = sorted([r for r in entries if r["group"] == "coupled_l2"], key=lambda x: x["strength"])
        wd_entries = sorted([r for r in entries if r["group"] == "decoupled_adamw"], key=lambda x: x["strength"])

        # fig08: Weight decay sweep (Coupled L2 vs Decoupled AdamW)
        lambdas = [0.0] + [r["strength"] for r in l2_entries]
        tr_l2 = [unreg["train_aggregate"]["mse_mean"]] + [r["train_aggregate"]["mse_mean"] for r in l2_entries]
        tr_l2_s = [unreg["train_aggregate"]["mse_std"]] + [r["train_aggregate"]["mse_std"] for r in l2_entries]
        va_l2 = [unreg["val_aggregate"]["mse_mean"]] + [r["val_aggregate"]["mse_mean"] for r in l2_entries]
        va_l2_s = [unreg["val_aggregate"]["mse_std"]] + [r["val_aggregate"]["mse_std"] for r in l2_entries]
        va_aw = [unreg["val_aggregate"]["mse_mean"]] + [r["val_aggregate"]["mse_mean"] for r in wd_entries]
        va_aw_s = [unreg["val_aggregate"]["mse_std"]] + [r["val_aggregate"]["mse_std"] for r in wd_entries]

        fig, ax = plt.subplots(figsize=(8.5, 5))
        xp = np.arange(len(lambdas))
        ax.errorbar(xp, tr_l2, yerr=tr_l2_s, fmt="o-", color=PALETTE["train"], capsize=3, label="Train MSE (Adam+L2)")
        ax.errorbar(xp, va_l2, yerr=va_l2_s, fmt="s--", color=PALETTE["val"], capsize=3, label="Val MSE (Adam+L2)")
        ax.errorbar(xp, va_aw, yerr=va_aw_s, fmt="D-.", color=PALETTE["emphasis"], capsize=3, label="Val MSE (AdamW)")
        ax.set_xticks(xp)
        ax.set_xticklabels([f"{l:.0e}" if l > 0 else "0" for l in lambdas])
        ax.set_xlabel("Penalty / Weight Decay Strength (lambda / wd)")
        ax.set_ylabel("MSE (" + ("units²" if dname == "synthetic" else "MPa²") + ")")
        ax.set_yscale("log")
        ax.set_title(f"E5: Weight Decay (Coupled L2 vs Decoupled AdamW) — {dname.title()}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fdir / f"fig08_weight_decay_{dname}.png")
        plt.close(fig)
        saved.append(fdir / f"fig08_weight_decay_{dname}.png")

        # fig09: Early stopping comparison
        es_entry = next(r for r in entries if r["label"] == "early_stop")
        best_l2_entry = min(l2_entries, key=lambda x: x["val_aggregate"]["mse_mean"])
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for idx, (lbl, entry) in enumerate([
            ("Unregularized", unreg),
            ("Early Stopping", es_entry),
            (f"Best L2 ({best_l2_entry['label']})", best_l2_entry),
        ]):
            sr0 = entry["seed_runs"][0]
            tr_c = sr0["train_mse_curve"]
            va_c = sr0["val_mse_curve"]
            if tr_c and va_c:
                c = _colour_cycle(idx)
                ep = np.arange(1, len(tr_c) + 1)
                ax.plot(ep, tr_c, color=c, lw=1.2, label=f"{lbl} train")
                ax.plot(ep, va_c, "--", color=c, lw=1.3, label=f"{lbl} val")
                if lbl == "Early Stopping" and sr0.get("stopped_epoch") is not None:
                    se = sr0["stopped_epoch"]
                    be = sr0.get("best_val_epoch", se)
                    ax.axvline(se, color=c, ls=":", lw=1.2, label=f"Stopped @ {se}")
                    ax.axvline(be, color="green", ls="-.", lw=1.2, label=f"Best ckpt @ {be}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (units²)" if dname == "synthetic" else "Standardized Training/Val MSE")
        ax.set_yscale("log")
        ax.set_title(f"E5: Early Stopping vs Unregularized & L2 — {dname.title()}")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(fdir / f"fig09_early_stopping_{dname}.png")
        plt.close(fig)
        saved.append(fdir / f"fig09_early_stopping_{dname}.png")

    # 10. fig10 from E6 Randomized Labels (N=50 & N=100 panels)
    for dname in ("synthetic", "concrete"):
        e6 = _load_json(ldir, f"E6_randomized_{dname}.json")
        if not e6:
            continue
        rows = e6["results"]
        widths = e6["widths"]
        unit_sq = "units²" if dname == "synthetic" else "MPa²"
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        n_focus = 100
        norm_rows = [next(r for r in rows if r["labels"] == "normal" and r["n_train"] == n_focus and r["hidden_width"] == h) for h in widths]
        rand_rows = [next(r for r in rows if r["labels"] == "random" and r["n_train"] == n_focus and r["hidden_width"] == h) for h in widths]

        norm_tr = np.array([r["train_aggregate"]["mse_mean"] for r in norm_rows])
        norm_tr_s = np.array([r["train_aggregate"]["mse_std"] for r in norm_rows])
        rand_tr = np.array([r["train_aggregate"]["mse_mean"] for r in rand_rows])
        rand_tr_s = np.array([r["train_aggregate"]["mse_std"] for r in rand_rows])

        ax = axes[0]
        ax.errorbar(widths, norm_tr,
                    yerr=[np.minimum(norm_tr_s, norm_tr * 0.7), norm_tr_s],
                    fmt="o-", color=PALETTE["normal"], capsize=3, label="Normal Train Targets")
        ax.errorbar(widths, rand_tr,
                    yerr=[np.minimum(rand_tr_s, rand_tr * 0.7), rand_tr_s],
                    fmt="s--", color=PALETTE["randomized"], capsize=3, label="Randomized Train Targets")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_yscale("log")
        ax.set_xlabel("Hidden Width H")
        ax.set_ylabel(f"Train MSE ({unit_sq})")
        ax.set_title(f"Training Error on Assigned Targets (N={n_focus})")
        ax.legend(fontsize=8)

        ax2 = axes[1]
        ax2.errorbar(widths, [r["val_aggregate"]["mse_mean"] for r in norm_rows],
                     yerr=[r["val_aggregate"]["mse_std"] for r in norm_rows],
                     fmt="o-", color=PALETTE["normal"], capsize=3, label="Trained on Normal Targets")
        ax2.errorbar(widths, [r["val_aggregate"]["mse_mean"] for r in rand_rows],
                     yerr=[r["val_aggregate"]["mse_std"] for r in rand_rows],
                     fmt="s--", color=PALETTE["randomized"], capsize=3, label="Trained on Randomized Targets")
        ax2.set_xscale("log", base=2)
        ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax2.set_yscale("log")
        ax2.set_xlabel("Hidden Width H")
        ax2.set_ylabel(f"Validation MSE ({unit_sq})")
        ax2.set_title("Validation Error on True Targets")
        ax2.legend(fontsize=8)

        fig.suptitle(f"E6: Memorization vs Generalization — {dname.title()}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(fdir / f"fig10_randomized_labels_{dname}.png")
        plt.close(fig)
        saved.append(fdir / f"fig10_randomized_labels_{dname}.png")

    # 11. fig11 from E7 Computational Budget
    for dname in ("synthetic", "concrete"):
        e7 = _load_json(ldir, f"E7_budget_{dname}.json")
        if not e7:
            continue
        pa_step = e7["protocol_a"]["fixed_step_results"]
        pa_flop = e7["protocol_a"]["matched_flop_results"]
        pb = e7["protocol_b"]["results"]
        widths = [r["hidden_width"] for r in pa_step]
        unit_sq = "units²" if dname == "synthetic" else "MPa²"

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        ax.plot(widths, [r["val_aggregate"]["mse_mean"] for r in pa_step], "o-", color=PALETTE["protocol_a"], label="Protocol A: Fixed Steps (2000)")
        ax.plot(widths, [r["val_aggregate"]["mse_mean"] for r in pa_flop], "^-.", color=PALETTE["test"], label="Protocol A: Matched FLOPs")
        ax.plot(widths, [r["val_aggregate"]["mse_mean"] for r in pb], "s--", color=PALETTE["protocol_b"], label="Protocol B: Convergence")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_yscale("log")
        ax.set_xlabel("Hidden Width H")
        ax.set_ylabel(f"Validation MSE ({unit_sq})")
        ax.set_title("Validation Performance Under Compute Protocols")
        ax.legend(fontsize=8)

        ax2 = axes[1]
        ax2.plot(widths, [r["total_flops"] / 1e6 for r in pa_step], "o-", color=PALETTE["protocol_a"], label="Fixed Steps (MFLOPs)")
        ax2.plot(widths, [r["total_flops"] / 1e6 for r in pa_flop], "^-.", color=PALETTE["test"], label="Matched FLOPs (MFLOPs)")
        ax2.plot(widths, [r["mean_flops"] / 1e6 for r in pb], "s--", color=PALETTE["protocol_b"], label="Protocol B (MFLOPs)")
        ax2.set_xscale("log", base=2)
        ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax2.set_yscale("log")
        ax2.set_xlabel("Hidden Width H")
        ax2.set_ylabel("Total Compute (Million FLOPs)")
        ax2.set_title("Computational Cost vs Model Width")
        ax2.legend(fontsize=8)

        fig.suptitle(f"E7: Computational Budget Protocols — {dname.title()}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(fdir / f"fig11_computational_budget_{dname}.png")
        plt.close(fig)
        saved.append(fdir / f"fig11_computational_budget_{dname}.png")

    # 12. fig12 from E8 Diagnostics
    for dname in ("synthetic", "concrete"):
        e8 = _load_json(ldir, f"E8_diagnostics_{dname}.json")
        if not e8:
            continue
        unit_sq = "units²" if dname == "synthetic" else "MPa²"
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

        # Panel 1: LR sweep (optimization failure)
        ax = axes[0]
        lr_rows = e8["lr_sweep"]["results"]
        xp = np.arange(len(lr_rows))
        ax.bar(xp - 0.18, [r["train_aggregate"]["mse_mean"] for r in lr_rows], 0.36, color=PALETTE["train"], label="Train")
        ax.bar(xp + 0.18, [r["val_aggregate"]["mse_mean"] for r in lr_rows], 0.36, color=PALETTE["val"], label="Val")
        ax.set_xticks(xp)
        ax.set_xticklabels([f"{r['lr']:.0e}" for r in lr_rows])
        ax.set_yscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel(f"MSE ({unit_sq})")
        ax.set_title("1. Optimization Failure (LR)")
        ax.legend(fontsize=8)

        # Panel 2: Overfitting interventions
        ax = axes[1]
        ov_rows = e8["overfitting_interventions"]["results"]
        xp = np.arange(len(ov_rows))
        ax.bar(xp - 0.18, [r["train_aggregate"]["mse_mean"] for r in ov_rows], 0.36, color=PALETTE["train"], label="Train")
        ax.bar(xp + 0.18, [r["val_aggregate"]["mse_mean"] for r in ov_rows], 0.36, color=PALETTE["val"], label="Val")
        ax.set_xticks(xp)
        ax.set_xticklabels([r["label"].replace("_H128_smallN", "").replace("_H128", "") for r in ov_rows], rotation=20, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel(f"MSE ({unit_sq})")
        ax.set_title("2. Overfitting Interventions")
        ax.legend(fontsize=8)

        # Panel 3: Inadequate representation interventions
        ax = axes[2]
        rep_rows = e8["representation_interventions"]["results"]
        xp = np.arange(len(rep_rows))
        ax.bar(xp - 0.18, [r["train_aggregate"]["mse_mean"] for r in rep_rows], 0.36, color=PALETTE["train"], label="Train")
        ax.bar(xp + 0.18, [r["val_aggregate"]["mse_mean"] for r in rep_rows], 0.36, color=PALETTE["val"], label="Val")
        ax.set_xticks(xp)
        ax.set_xticklabels([r["label"].replace("_baseline", "").replace("mlp_", "") for r in rep_rows], rotation=20, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel(f"MSE ({unit_sq})")
        ax.set_title("3. Inadequate Representation")
        ax.legend(fontsize=8)

        fig.suptitle(f"E8: Controlled Diagnostic Interventions — {dname.title()}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(fdir / f"fig12_diagnostics_{dname}.png")
        plt.close(fig)
        saved.append(fdir / f"fig12_diagnostics_{dname}.png")

    # 14. fig14: Final comparison of selected models on untouched test set
    e2 = _load_json(ldir, "E2_baseline.json")
    e3_syn = _load_json(ldir, "E3_capacity_synthetic.json")
    e3_con = _load_json(ldir, "E3_capacity_concrete.json")
    e5_syn = _load_json(ldir, "E5_regularization_synthetic.json")
    e5_con = _load_json(ldir, "E5_regularization_concrete.json")
    if e2 and e3_syn and e3_con and e5_syn and e5_con:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, dname, e3, e5, unit in [
            (axes[0], "synthetic", e3_syn, e5_syn, "units"),
            (axes[1], "concrete", e3_con, e5_con, "MPa"),
        ]:
            lin_m = e2["aggregates"][f"{dname}_linear"]["test"]
            best_h_m = e3["selected_test_aggregate"]
            pt = e5["post_selection_test"]
            unreg_m = pt["unreg"]
            es_m = pt["early_stop"]
            best_l2_lbl = e5["best_l2_label"]
            best_l2_m = pt[best_l2_lbl]
            best_aw_lbl = e5["best_adamw_label"]
            best_aw_m = pt[best_aw_lbl]

            items = [
                ("Linear\nOLS", lin_m),
                (f"Best Width\n(H={e3['best_H']})", best_h_m),
                ("H=128\nUnreg", unreg_m),
                (f"H=128\n{best_l2_lbl.replace('l2_lambda=', 'L2=')}", best_l2_m),
                (f"H=128\n{best_aw_lbl.replace('adamw_wd=', 'AdamW=')}", best_aw_m),
                ("H=128\nEarlyStop", es_m),
            ]
            names = [x[0] for x in items]
            rmses = [x[1]["rmse_mean"] for x in items]
            stds = [x[1]["rmse_std"] for x in items]
            r2s = [x[1]["r2_mean"] for x in items]

            xp = np.arange(len(names))
            bars = ax.bar(xp, rmses, yerr=stds, capsize=3, color=[_colour_cycle(i) for i in range(len(names))], edgecolor="k", lw=0.6)
            for b, sd, r2 in zip(bars, stds, r2s):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + sd + 0.02 * max(rmses), f"R²={r2:.3f}", ha="center", va="bottom", fontsize=7.5)
            ax.set_ylim(0, max(r + s for r, s in zip(rmses, stds)) * 1.13)
            ax.set_xticks(xp)
            ax.set_xticklabels(names, fontsize=8)
            ax.set_ylabel(f"Test RMSE ({unit})")
            ax.set_title(f"{dname.title()} Untouched Test Performance")
        fig.suptitle("Final Model Comparison on Untouched Held-Out Test Sets", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(fdir / "fig14_final_comparison.png")
        plt.close(fig)
        saved.append(fdir / "fig14_final_comparison.png")

    # 15-18. Extension figures from E9_extensions.json
    e9 = _load_json(ldir, "E9_extensions.json")
    if e9:
        # fig15: E9A Warmup ablation
        e9a = e9.get("E9A_warmup", {}).get("results", [])
        if e9a:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            lbls = [r["label"] for r in e9a]
            vals = np.array([r["val_aggregate"]["mse_mean"] for r in e9a])
            stds = np.array([r["val_aggregate"]["mse_std"] for r in e9a])
            xp = np.arange(len(lbls))
            bars = ax.bar(xp, vals, yerr=[np.minimum(stds, vals * 0.7), stds], capsize=4,
                          color=[_colour_cycle(i) for i in range(len(lbls))], edgecolor="k", lw=0.6)
            for b, sd in zip(bars, stds):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + sd + 0.03,
                        f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(xp)
            ax.set_xticklabels(lbls)
            ax.set_ylabel("Validation MSE (units²)")
            ax.set_title("E9A: Early-Stopping Warmup Ablation (Synthetic N=50, H=128)")
            fig.tight_layout()
            fig.savefig(fdir / "fig15_ext_warmup_ablation.png")
            plt.close(fig)
            saved.append(fdir / "fig15_ext_warmup_ablation.png")

        # fig16: E9B Mini-batch SGD vs Adam on Concrete
        e9b = e9.get("E9B_minibatch", {}).get("results", [])
        if e9b:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
            batch_labels = ["16", "32", "64", "full(618)"]
            for ax, opt_name in [(axes[0], "momentum"), (axes[1], "adam")]:
                for idx, H in enumerate([64, 128, 256]):
                    sub = [next(r for r in e9b if r["optimizer"] == opt_name and r["hidden_width"] == H and r["batch_label"] == bl) for bl in batch_labels]
                    ax.errorbar(range(len(batch_labels)), [r["val_aggregate"]["mse_mean"] for r in sub],
                                yerr=[r["val_aggregate"]["mse_std"] for r in sub],
                                fmt="o-", capsize=3, color=_colour_cycle(idx), label=f"H={H}")
                ax.set_xticks(range(len(batch_labels)))
                ax.set_xticklabels(batch_labels)
                ax.set_xlabel("Batch Size")
                ax.set_title(f"Optimizer: {opt_name.upper()}")
                ax.legend(fontsize=8)
            axes[0].set_ylabel("Validation MSE (MPa²)")
            fig.suptitle("E9B: Mini-Batch Implicit Regularization on Concrete", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fig.savefig(fdir / "fig16_ext_minibatch_sgd.png")
            plt.close(fig)
            saved.append(fdir / "fig16_ext_minibatch_sgd.png")

        # fig17: E9C 5-Fold CV on Concrete
        e9c = e9.get("E9C_kfold", {}).get("results", [])
        if e9c:
            fig, ax = plt.subplots(figsize=(9, 5))
            names = [r["model"].replace(" ", "\n", 1) for r in e9c]
            rmses = [r["across_fold_aggregate"]["rmse_mean"] for r in e9c]
            stds = [r["across_fold_aggregate"]["rmse_std"] for r in e9c]
            r2s = [r["across_fold_aggregate"]["r2_mean"] for r in e9c]
            xp = np.arange(len(names))
            bars = ax.bar(xp, rmses, yerr=stds, capsize=4, color=[_colour_cycle(i) for i in range(len(names))], edgecolor="k", lw=0.6)
            for b, sd, r2 in zip(bars, stds, r2s):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + sd + 0.15,
                        f"{b.get_height():.2f} MPa\n(R²={r2:.3f})", ha="center", va="bottom", fontsize=8)
            ax.set_ylim(0, max(r + s for r, s in zip(rmses, stds)) * 1.18)
            ax.set_xticks(xp)
            ax.set_xticklabels(names, fontsize=8.5)
            ax.set_ylabel("5-Fold Outer Test RMSE (MPa)")
            ax.set_title("E9C: 5-Fold Cross-Validation on UCI Concrete (Mean ± Split Std)")
            fig.tight_layout()
            fig.savefig(fdir / "fig17_ext_kfold_concrete.png")
            plt.close(fig)
            saved.append(fdir / "fig17_ext_kfold_concrete.png")

        # fig18: E9D Double Descent & 2-HL Depth Comparison
        e9d = e9.get("E9D_double_descent", {})
        one_hl = e9d.get("one_hidden_layer_results", [])
        two_hl = e9d.get("two_hidden_layer_results", [])
        if one_hl:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            pn = np.array([r["p_over_n"] for r in one_hl])
            tr_m = np.array([r["train_aggregate"]["mse_mean"] for r in one_hl])
            val_m = np.array([r["val_aggregate"]["mse_mean"] for r in one_hl])
            test_m = np.array([r["test_aggregate"]["mse_mean"] for r in one_hl])
            val_s = np.array([r["val_aggregate"]["mse_std"] for r in one_hl])

            ax = axes[0]
            ax.plot(pn, tr_m, "o-", color=PALETTE["train"], label="1-HL Train MSE")
            ax.errorbar(pn, val_m, yerr=val_s, fmt="s--", capsize=3, color=PALETTE["val"], label="1-HL Val MSE")
            ax.plot(pn, test_m, "^:", color=PALETTE["test"], label="1-HL Test MSE")
            ax.axvline(1.0, color="gray", ls=":", lw=1.2, label="Interpolation (P/N=1)")
            if two_hl:
                ax.scatter([r["p_over_n"] for r in two_hl], [r["test_aggregate"]["mse_mean"] for r in two_hl],
                           marker="D", s=65, color=PALETTE["emphasis"], zorder=6, label="2-HL Test MSE")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Overparameterization Ratio P / N")
            ax.set_ylabel("MSE (MPa²)")
            ax.set_title("Double-Descent Curve (H = 4 .. 1024)")
            ax.legend(fontsize=8)

            ax2 = axes[1]
            widths_lbl = [f"H={r['hidden_width']}\n(P={r['n_params']})" for r in one_hl]
            xp = np.arange(len(one_hl))
            ax2.bar(xp, [r["test_aggregate"]["r2_mean"] for r in one_hl], color=PALETTE["nn"], alpha=0.85, edgecolor="k", lw=0.5, label="1-HL Test R²")
            ax2.set_xticks(xp)
            ax2.set_xticklabels(widths_lbl, rotation=30, ha="right", fontsize=7.5)
            ax2.set_ylabel("Test R²")
            ax2.set_ylim(0.75, 0.95)
            ax2.set_title("Test R² Across Extreme Overparameterization")
            ax2.legend(fontsize=8)

            fig.suptitle("E9D: Extreme Overparameterization (P/N up to 16.57) & Depth Comparison", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fig.savefig(fdir / "fig18_ext_double_descent.png")
            plt.close(fig)
            saved.append(fdir / "fig18_ext_double_descent.png")

    print(f"[Plots] Generated {len(saved)} figures in {fdir}")
    return saved


if __name__ == "__main__":
    generate_all_figures()
