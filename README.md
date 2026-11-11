# Project 9: Neural-Network Capacity and Generalization

A rigorous, reproducible investigation of capacity, generalization, overfitting, regularization, memorization, and optimization diagnostics in one-hidden-layer neural networks (plus extended mini-batch, 5-fold CV, warmup, and double-descent analyses) on synthetic nonlinear regression and the UCI Concrete Compressive Strength benchmark.

## Repository Structure

```text
project-9/
├── config/
│   └── experiment_config.py        # Shared configs, hyperparameter grids, seeds
├── data/
│   ├── raw/concrete.csv            # UCI Concrete Compressive Strength (1030 x 9)
│   └── processed/                  # Deterministic split manifest & metadata
├── src/
│   ├── data/                       # Synthetic generator & leakage-free Concrete (single + 5-fold) loader
│   ├── models/                     # Pure-NumPy MLP, PyTorch 1-HL & 2-HL MLPs, OLS/Ridge baseline
│   ├── training/                   # Backprop helpers, optimizers, L2/WD, warmup early stopping, trainer
│   ├── experiments/                # E1–E9 modular experiment runners
│   ├── evaluation/                 # Physical-unit metrics, multi-seed aggregation, diagnostics, CSV exporter
│   └── visualization/              # Publication-quality figure generator (fig01–fig18, 27 PNGs)
├── tests/
│   └── test_pipeline.py            # 9 unit tests: gradients, leakage, k-fold, warmup, determinism, PyTorch match
├── results/
│   ├── experiment_logs/            # 15 standardized JSON run outputs (E1–E9)
│   ├── tables/                     # 17 aggregated CSV & Markdown tables
│   └── figures/                    # 27 300-DPI PNG figures
├── report/
│   └── project_report.md           # Full technical research report with Section 19B extensions
├── presentation/
│   ├── build_presentation.py       # Dynamic 20-slide PPTX builder from live logs & figures
│   ├── project9_presentation.pptx  # 20-slide deck with real experimental plots
│   ├── speaker_notes.md            # Six-member speaker guide
│   └── viva_qa_bank.md             # Viva examination Q&A bank
└── scripts/
    ├── run_all_experiments.py      # End-to-end pipeline orchestrator with CLI flags
    ├── inspect_results.py          # Compact CLI summary printer for E1–E9 JSON logs
    └── audit_compliance.py         # Automated Phase 12 reproducibility & leakage audit
```

## Setup & Reproduction

```bash
pip install -r requirements.txt
python tests/test_pipeline.py
```

### Running the Pipeline
- **Full end-to-end pipeline (`E1`–`E9` + tables + figures + audit):**
  ```bash
  python scripts/run_all_experiments.py
  ```
- **Core `E1`–`E8` pipeline only (skips ~30 min `E9` high-width PyTorch/5-fold sweeps):**
  ```bash
  python scripts/run_all_experiments.py --skip-extensions
  ```
- **Targeted experiments or post-processing only:**
  ```bash
  python scripts/run_all_experiments.py --only E1 E2 --skip-tests
  python scripts/run_all_experiments.py --post-only
  ```
- **Inspect results & run compliance audit directly:**
  ```bash
  python scripts/inspect_results.py
  python scripts/audit_compliance.py
  ```

### Data Acquisition
- **Synthetic dataset:** Generated deterministically from `f*(x) = 2 sin(2x) + 0.5 x^2 + eps`, `x ~ U(-2.5, 2.5)`, `eps ~ N(0, 0.35^2)` (irreducible Bayes MSE = `0.1225`).
- **UCI Concrete Compressive Strength:** Cached in `data/raw/concrete.csv` (SHA-256 `d7d8bd087f832935e902bcb2687667238cac3fe06799677a93ca2f46dce8db02`, 1030 samples, 8 features, target in MPa). Scalers are fitted strictly on active training splits (and per fold in 5-fold CV); the test set remains untouched during model selection.
