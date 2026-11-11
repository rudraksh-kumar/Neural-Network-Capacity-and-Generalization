"""Controlled nonlinear synthetic regression dataset generator."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from config.experiment_config import SyntheticDataConfig


def ground_truth_function(x: np.ndarray) -> np.ndarray:
    """Population nonlinear relationship f*(x) = 2*sin(2x) + 0.5*x^2."""
    x_arr = np.asarray(x, dtype=np.float64)
    return 2.0 * np.sin(2.0 * x_arr) + 0.5 * (x_arr ** 2)


@dataclass
class DatasetSplit:
    """Container for split arrays and metadata without any test-set leakage."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_train_original: np.ndarray
    f_train_clean: np.ndarray
    f_val_clean: np.ndarray
    f_test_clean: np.ndarray
    labels_randomized: bool
    feature_names: list[str]
    target_name: str
    target_unit: str
    irreducible_mse: float | None = None

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Identity inverse transform for synthetic datasets (no target scaling)."""
        return np.asarray(y_scaled, dtype=np.float64)


def _generate_split(
    n_samples: int,
    cfg: SyntheticDataConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (X, y_noisy, f_clean) independently for a single split."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(cfg.x_min, cfg.x_max, size=(n_samples, 1))
    f_clean = ground_truth_function(X)
    noise = rng.normal(loc=0.0, scale=cfg.noise_std, size=(n_samples, 1))
    y = f_clean + noise
    return X.astype(np.float64), y.astype(np.float64), f_clean.astype(np.float64)


def build_synthetic_dataset(
    cfg: SyntheticDataConfig | None = None,
    n_train: int | None = None,
    randomize_train_labels: bool = False,
    train_seed: int | None = None,
    label_perm_seed: int | None = None,
) -> DatasetSplit:
    """Build synthetic train/val/test splits with fixed population law.

    Validation and test splits remain completely invariant and unpermuted across
    all training sample sizes and randomized-label experiments.
    """
    cfg = cfg or SyntheticDataConfig()
    n_tr = cfg.n_train_default if n_train is None else int(n_train)
    tr_seed = cfg.train_seed if train_seed is None else int(train_seed)
    perm_seed = cfg.label_perm_seed if label_perm_seed is None else int(label_perm_seed)

    X_train, y_train_orig, f_train = _generate_split(n_tr, cfg, tr_seed)
    X_val, y_val, f_val = _generate_split(cfg.n_val, cfg, cfg.val_seed)
    X_test, y_test, f_test = _generate_split(cfg.n_test, cfg, cfg.test_seed)

    if randomize_train_labels:
        perm_rng = np.random.default_rng(perm_seed)
        perm_idx = perm_rng.permutation(n_tr)
        y_train = y_train_orig[perm_idx].copy()
    else:
        y_train = y_train_orig.copy()

    return DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        y_train_original=y_train_orig,
        f_train_clean=f_train,
        f_val_clean=f_val,
        f_test_clean=f_test,
        labels_randomized=randomize_train_labels,
        feature_names=["x"],
        target_name="y",
        target_unit="units",
        irreducible_mse=cfg.irreducible_mse,
    )

