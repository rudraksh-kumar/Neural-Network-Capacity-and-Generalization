"""Automated Phase 12 reproducibility and compliance audit script."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.experiment_config import ProjectPaths
from src.data.real_data import load_concrete_dataset
from src.data.synthetic_data import build_synthetic_dataset


REQUIRED_LOGS = [
    "E1_gradient_check.json",
    "E2_baseline.json",
    "E3_capacity_synthetic.json",
    "E3_capacity_concrete.json",
    "E4_sample_size_synthetic.json",
    "E4_sample_size_concrete.json",
    "E5_regularization_synthetic.json",
    "E5_regularization_concrete.json",
    "E6_randomized_synthetic.json",
    "E6_randomized_concrete.json",
    "E7_budget_synthetic.json",
    "E7_budget_concrete.json",
    "E8_diagnostics_synthetic.json",
    "E8_diagnostics_concrete.json",
    "E9_extensions.json",
]

REQUIRED_TABLES = [
    "table_e1_gradient_verification.csv",
    "table_e2_baseline.csv",
    "table_e3_capacity_concrete.csv",
    "table_e3_capacity_synthetic.csv",
    "table_e4_sample_size_concrete.csv",
    "table_e4_sample_size_synthetic.csv",
    "table_e5_regularization_concrete.csv",
    "table_e5_regularization_synthetic.csv",
    "table_e6_randomized_concrete.csv",
    "table_e6_randomized_synthetic.csv",
    "table_e7_budget_concrete.csv",
    "table_e7_budget_synthetic.csv",
    "table_e8_diagnostics.md",
    "table_e9a_warmup.csv",
    "table_e9b_minibatch.csv",
    "table_e9c_kfold.csv",
    "table_e9d_double_descent.csv",
]

REQUIRED_FIGURES = [
    "fig01_synthetic_ground_truth.png",
    "fig02_gradient_verification.png",
    "fig03_linear_vs_nn.png",
    "fig04_training_curves_synthetic.png",
    "fig04_training_curves_concrete.png",
    "fig05_width_vs_performance_synthetic.png",
    "fig05_width_vs_performance_concrete.png",
    "fig06_sample_size_synthetic.png",
    "fig06_sample_size_concrete.png",
    "fig07_generalization_gap_synthetic.png",
    "fig07_generalization_gap_concrete.png",
    "fig08_weight_decay_synthetic.png",
    "fig08_weight_decay_concrete.png",
    "fig09_early_stopping_synthetic.png",
    "fig09_early_stopping_concrete.png",
    "fig10_randomized_labels_synthetic.png",
    "fig10_randomized_labels_concrete.png",
    "fig11_computational_budget_synthetic.png",
    "fig11_computational_budget_concrete.png",
    "fig12_diagnostics_synthetic.png",
    "fig12_diagnostics_concrete.png",
    "fig13_real_predictions.png",
    "fig14_final_comparison.png",
    "fig15_ext_warmup_ablation.png",
    "fig16_ext_minibatch_sgd.png",
    "fig17_ext_kfold_concrete.png",
    "fig18_ext_double_descent.png",
]


def _check_e6_unpermuted_val_test(paths: ProjectPaths) -> tuple[bool, str]:
    """Verify E6 keeps validation & test targets untouched during label permutation."""
    for fn in ("E6_randomized_synthetic.json", "E6_randomized_concrete.json"):
        p = paths.results_logs / fn
        if not p.exists():
            return False, f"Missing {fn}"
        data = json.load(open(p, "r", encoding="utf-8"))
        modes = {r.get("labels") for r in data.get("results", [])}
        if not {"normal", "random"}.issubset(modes):
            return False, f"{fn} missing normal/random split modes: {modes}"

    syn_n = build_synthetic_dataset(randomize_train_labels=False)
    syn_r = build_synthetic_dataset(randomize_train_labels=True)
    if not (
        np.array_equal(syn_n.y_val, syn_r.y_val)
        and np.array_equal(syn_n.y_test, syn_r.y_test)
        and not np.array_equal(syn_n.y_train, syn_r.y_train)
    ):
        return False, "Synthetic val/test leaked or train not permuted"

    con_n = load_concrete_dataset(randomize_train_labels=False)
    con_r = load_concrete_dataset(randomize_train_labels=True)
    if not (
        np.allclose(con_n.inverse_transform_y(con_n.y_val), con_r.inverse_transform_y(con_r.y_val))
        and np.allclose(con_n.inverse_transform_y(con_n.y_test), con_r.inverse_transform_y(con_r.y_test))
        and not np.array_equal(con_n.y_train, con_r.y_train)
    ):
        return False, "Concrete val/test leaked or train not permuted"

    return True, "Synthetic & Concrete val/test strictly unpermuted"


def run_audit() -> bool:
    paths = ProjectPaths()
    checks: list[tuple[str, bool, str]] = []

    # 1. Experiment logs
    missing_logs = [f for f in REQUIRED_LOGS if not (paths.results_logs / f).exists()]
    checks.append((f"All {len(REQUIRED_LOGS)} experiment JSON logs present", len(missing_logs) == 0, f"missing={missing_logs}"))

    # 2. Gradient check verdict
    e1_path = paths.results_logs / "E1_gradient_check.json"
    e1_ok = False
    if e1_path.exists():
        e1 = json.load(open(e1_path, "r", encoding="utf-8"))
        e1_ok = bool(e1.get("all_passed", False))
    checks.append(("E1 gradient verification passed", e1_ok, "FD + PyTorch autograd"))

    # 3. No test leakage during E3 & E5 selection
    no_leak = True
    for fn, list_key in [
        ("E3_capacity_synthetic.json", "width_results"),
        ("E3_capacity_concrete.json", "width_results"),
        ("E5_regularization_synthetic.json", "reg_results"),
        ("E5_regularization_concrete.json", "reg_results"),
    ]:
        p = paths.results_logs / fn
        if p.exists():
            data = json.load(open(p, "r", encoding="utf-8"))
            for item in data[list_key]:
                for sr in item["seed_runs"]:
                    if "test_metrics" in sr:
                        no_leak = False
        else:
            no_leak = False
    checks.append(("Zero test evaluation inside E3/E5 selection loops", no_leak, "Verified"))

    # 4. E6 unpermuted validation/test targets check
    e6_ok, e6_detail = _check_e6_unpermuted_val_test(paths)
    checks.append(("E6 validation & test targets unpermuted", e6_ok, e6_detail))

    # 5. Exported tables
    missing_tables = [
        t for t in REQUIRED_TABLES
        if not (paths.results_tables / t).exists() or (paths.results_tables / t).stat().st_size < 50
    ]
    checks.append((f"All {len(REQUIRED_TABLES)} exported tables present", len(missing_tables) == 0, f"missing={missing_tables}"))

    # 6. All figures
    missing_figs = [
        f for f in REQUIRED_FIGURES
        if not (paths.results_figures / f).exists() or (paths.results_figures / f).stat().st_size < 5000
    ]
    checks.append((f"All {len(REQUIRED_FIGURES)} required figures generated (>5KB)", len(missing_figs) == 0, f"missing={missing_figs}"))

    # 7. Report & presentation deliverables
    rep_ok = (paths.report_dir / "project_report.md").exists() and (paths.report_dir / "project_report.md").stat().st_size > 5000
    pptx_ok = (paths.presentation_dir / "project9_presentation.pptx").exists() and (paths.presentation_dir / "project9_presentation.pptx").stat().st_size > 20000
    notes_ok = (paths.presentation_dir / "speaker_notes.md").exists() and (paths.presentation_dir / "viva_qa_bank.md").exists()
    checks.append(("Report markdown present", rep_ok, str(paths.report_dir / "project_report.md")))
    checks.append(("Presentation PPTX present", pptx_ok, str(paths.presentation_dir / "project9_presentation.pptx")))
    checks.append(("Speaker notes & Viva QA bank present", notes_ok, str(paths.presentation_dir)))

    print("\n=== PHASE 12 COMPLIANCE AUDIT ===")
    all_ok = True
    for label, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label:<54s} ({detail})")
        if not ok:
            all_ok = False
    print(f"FINAL AUDIT: {'PASS' if all_ok else 'FAIL'}\n")
    return all_ok


if __name__ == "__main__":
    ok = run_audit()
    sys.exit(0 if ok else 1)
