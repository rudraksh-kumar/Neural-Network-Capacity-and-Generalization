"""Master end-to-end pipeline orchestrator for Project 9.

Executes E1–E9 in deterministic order (with optional CLI filters), exports
CSV/Markdown summary tables, generates all publication figures, and runs the
final Phase 12 compliance audit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.experiment_config import ProjectPaths
from scripts.audit_compliance import run_audit
from src.evaluation.diagnostics import generate_diagnostic_summary
from src.evaluation.export_tables import export_all_tables
from src.experiments import (
    baseline_experiment,
    capacity_experiment,
    computational_budget,
    extensions_experiment,
    gradient_check,
    randomized_labels,
    regularization_experiment,
    sample_size_experiment,
)
from src.visualization.generate_figures import generate_all_figures


EXPERIMENT_STEPS = [
    ("E1", "Finite-Difference & PyTorch Gradient Verification", gradient_check.run_experiment),
    ("E2", "Linear Baseline vs One-Hidden-Layer MLP", baseline_experiment.run_experiment),
    ("E3", "Hidden-Layer Width Capacity Sweep", capacity_experiment.run_experiment),
    ("E4", "Training-Set Size Scaling Sweep", sample_size_experiment.run_experiment),
    ("E5", "Regularization, Weight Decay & Early Stopping", regularization_experiment.run_experiment),
    ("E6", "Randomized-Label Memorization vs Generalization", randomized_labels.run_experiment),
    ("E7_E8", "Computational Budgets & Controlled Diagnostics", computational_budget.run_experiment),
    ("E9", "Scientific Extensions (Warmup, Mini-Batch, 5-Fold CV, Double Descent)", extensions_experiment.run_experiment),
]


def _resolve_selected_steps(only: list[str] | None, skip_extensions: bool, post_only: bool) -> list[tuple]:
    if post_only:
        return []
    valid_ids = {eid for eid, _, _ in EXPERIMENT_STEPS}
    if only:
        normalized = set()
        for raw in only:
            tok = raw.strip().upper()
            if tok in ("E7", "E8"):
                normalized.add("E7_E8")
            elif tok in valid_ids:
                normalized.add(tok)
            else:
                raise ValueError(f"Unknown experiment ID '{raw}'. Valid choices: E1..E9 (or E7_E8).")
        selected = [item for item in EXPERIMENT_STEPS if item[0] in normalized]
    else:
        selected = list(EXPERIMENT_STEPS)

    if skip_extensions:
        selected = [item for item in selected if item[0] != "E9"]
    return selected


def run_all(
    skip_tests: bool = False,
    skip_extensions: bool = False,
    only: list[str] | None = None,
    post_only: bool = False,
    skip_audit: bool = False,
) -> None:
    t0 = time.perf_counter()
    paths = ProjectPaths()
    paths.ensure_dirs()

    if not skip_tests:
        print("\n=== STEP 0: Running Unit & Data-Leakage Verification Tests ===")
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=str(paths.root / "tests"), pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        res = runner.run(suite)
        if not res.wasSuccessful():
            raise RuntimeError("Unit tests failed; aborting pipeline.")

    selected = _resolve_selected_steps(only=only, skip_extensions=skip_extensions, post_only=post_only)
    for idx, (exp_id, title, fn) in enumerate(selected, start=1):
        print(f"\n=== STEP {idx}: {exp_id} — {title} ===")
        out = fn()
        if exp_id == "E1" and not out["all_passed"]:
            raise RuntimeError("Gradient verification failed!")

    print("\n=== EXPORTING TABLES, DIAGNOSTICS & FIGURES ===")
    diag = generate_diagnostic_summary(paths.results_logs)
    if diag["rows"]:
        print(diag["markdown_table"])
    export_all_tables(paths.results_logs, paths.results_tables)
    figs = generate_all_figures(paths.results_logs, paths.results_figures)

    if not skip_audit:
        audit_ok = run_audit()
        if not audit_ok:
            raise RuntimeError("Phase 12 compliance audit failed!")

    elapsed = time.perf_counter() - t0
    print(f"\n=== PIPELINE COMPLETE in {elapsed:.1f}s ({len(figs)} figures generated) ===")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Project 9 experiments (E1–E9), export tables/figures, and audit compliance."
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests/test_pipeline.py before experiments.",
    )
    parser.add_argument(
        "--skip-extensions",
        action="store_true",
        help="Run core E1–E8 experiments and skip the ~30-minute E9 extension suite.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="EXP",
        help="Run only specified experiments (e.g., --only E1 E2 or --only E7 E8 E9).",
    )
    parser.add_argument(
        "--post-only",
        action="store_true",
        help="Skip running experiments; only regenerate tables, figures, and run audit.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip the final Phase 12 compliance audit.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_all(
        skip_tests=args.skip_tests,
        skip_extensions=args.skip_extensions,
        only=args.only,
        post_only=args.post_only,
        skip_audit=args.skip_audit,
    )
