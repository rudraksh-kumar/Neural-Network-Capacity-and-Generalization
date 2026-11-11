"""Centralized experiment configurations, grids, and seeds for Project 9."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical project filesystem paths."""

    root: Path = ROOT_DIR
    data_raw: Path = ROOT_DIR / "data" / "raw"
    data_processed: Path = ROOT_DIR / "data" / "processed"
    results_tables: Path = ROOT_DIR / "results" / "tables"
    results_figures: Path = ROOT_DIR / "results" / "figures"
    results_logs: Path = ROOT_DIR / "results" / "experiment_logs"
    report_dir: Path = ROOT_DIR / "report"
    presentation_dir: Path = ROOT_DIR / "presentation"

    def ensure_dirs(self) -> None:
        """Create output and data directories if absent."""
        for path in (
            self.data_raw,
            self.data_processed,
            self.results_tables,
            self.results_figures,
            self.results_logs,
            self.report_dir,
            self.presentation_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Configuration for the controlled 1D nonlinear regression dataset."""

    x_min: float = -2.5
    x_max: float = 2.5
    noise_std: float = 0.35
    n_train_default: int = 100
    n_val: int = 1000
    n_test: int = 2000
    train_seed: int = 1001
    val_seed: int = 2002
    test_seed: int = 3003
    label_perm_seed: int = 4004

    @property
    def irreducible_mse(self) -> float:
        """Population Bayes-optimal MSE sigma^2."""
        return float(self.noise_std ** 2)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["irreducible_mse"] = self.irreducible_mse
        return d


@dataclass(frozen=True)
class RealDataConfig:
    """Configuration for the UCI Concrete Compressive Strength benchmark."""

    dataset_name: str = "uci_concrete"
    raw_csv_path: str = str(ROOT_DIR / "data" / "raw" / "concrete.csv")
    split_ratio_train: float = 0.60
    split_ratio_val: float = 0.20
    split_ratio_test: float = 0.20
    split_seed: int = 42
    subsample_seed: int = 777
    label_perm_seed: int = 8888
    standardize_x: bool = True
    standardize_y_for_training: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and initialization configuration for one-hidden-layer MLP."""

    input_dim: int = 1
    hidden_width: int = 32
    activation: str = "tanh"  # 'tanh' or 'relu'
    init_scheme: str = "auto"  # 'auto', 'xavier', 'he', 'normal'
    init_scale: float | None = None
    dtype: str = "float64"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    """Optimization, regularization, early stopping, and compute budget settings."""

    optimizer: str = "adam"  # 'gd', 'momentum', 'adam', 'adamw'
    learning_rate: float = 0.01
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    batch_size: int | None = None  # None -> full-batch training
    max_epochs: int = 3000
    l2_lambda: float = 0.0
    weight_decay: float = 0.0
    regularize_biases: bool = False
    early_stopping: bool = False
    patience: int = 60
    min_delta: float = 1e-5
    warmup_epochs: int = 0
    restore_best: bool = True
    max_steps: int | None = None
    max_flops: float | None = None
    convergence_tol: float | None = None
    convergence_patience: int = 50
    shuffle_batches: bool = True
    eval_val_every: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Canonical experimental grids
WIDTH_GRID: list[int] = [4, 8, 16, 32, 64, 128, 256]
EXTENDED_WIDTH_GRID: list[int] = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
BATCH_SIZE_GRID: list[int | None] = [16, 32, 64, None]
WARMUP_GRID: list[int] = [0, 100, 250, 500]
SAMPLE_SIZE_GRID_SYN: list[int] = [25, 50, 100, 200, 400, 800]
SAMPLE_SIZE_GRID_REAL: list[int] = [25, 50, 100, 200, 400, 618]
WEIGHT_DECAY_GRID: list[float] = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

INIT_SEEDS_SYN: list[int] = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
INIT_SEEDS_REAL: list[int] = [201, 202, 203, 204, 205]

# Pilot / default controlled settings
DEFAULT_SYN_TRAIN_CFG = TrainConfig(
    optimizer="adam",
    learning_rate=0.01,
    batch_size=None,
    max_epochs=2500,
)

DEFAULT_REAL_TRAIN_CFG = TrainConfig(
    optimizer="adam",
    learning_rate=0.01,
    batch_size=None,
    max_epochs=2500,
)

