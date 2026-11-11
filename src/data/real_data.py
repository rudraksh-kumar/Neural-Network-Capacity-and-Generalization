"""UCI Concrete Compressive Strength dataset loader with strict leakage-free splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.experiment_config import RealDataConfig, ProjectPaths


@dataclass
class RealDatasetSplit:
    """Container for real-world train/val/test arrays with inverse-transform capability."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_train_original: np.ndarray  # before any label permutation
    labels_randomized: bool
    feature_names: list[str]
    target_name: str
    target_unit: str
    n_features: int
    n_total: int
    x_scaler: StandardScaler
    y_scaler: StandardScaler | None
    irreducible_mse: float | None  # unknown for real data

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original MPa units."""
        y_arr = np.asarray(y_scaled, dtype=np.float64).reshape(-1, 1)
        if self.y_scaler is not None:
            return self.y_scaler.inverse_transform(y_arr).reshape(-1, 1)
        return y_arr


CONCRETE_FEATURE_DESCRIPTIONS = {
    "cement": "Cement (kg/m³)",
    "slag": "Blast Furnace Slag (kg/m³)",
    "ash": "Fly Ash (kg/m³)",
    "water": "Water (kg/m³)",
    "superplastic": "Superplasticizer (kg/m³)",
    "coarseagg": "Coarse Aggregate (kg/m³)",
    "fineagg": "Fine Aggregate (kg/m³)",
    "age": "Age (days)",
}


def load_concrete_dataset(
    cfg: RealDataConfig | None = None,
    n_train_subsample: int | None = None,
    randomize_train_labels: bool = False,
    subsample_seed: int | None = None,
    label_perm_seed: int | None = None,
) -> RealDatasetSplit:
    """Load, split, and preprocess the Concrete dataset with no data leakage.

    Parameters
    ----------
    cfg : RealDataConfig
        Dataset configuration (paths, split ratios, seeds, scaling flags).
    n_train_subsample : int | None
        If given, randomly subsample the training set to this size *after* the
        full 60/20/20 split but *before* fitting the scaler, so each subsample
        size gets its own scaler fitted only on its active training observations.
    randomize_train_labels : bool
        Independently permute training targets (preserving inputs & val/test).
    subsample_seed : int | None
        Override seed for subsampling.
    label_perm_seed : int | None
        Override seed for label permutation.
    """
    cfg = cfg or RealDataConfig()
    paths = ProjectPaths()
    csv_path = Path(cfg.raw_csv_path)
    if not csv_path.exists():
        csv_path = paths.data_raw / "concrete.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Concrete CSV not found at {csv_path}. "
            "Run the download step first (see README.md)."
        )

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "strength"]
    target_col = "strength"

    X_all = df[feature_cols].values.astype(np.float64)
    y_all = df[[target_col]].values.astype(np.float64)
    n_total = len(df)

    # --- Deterministic three-way split (indices only) ---
    rng_split = np.random.default_rng(cfg.split_seed)
    indices = rng_split.permutation(n_total)

    n_test = int(np.round(cfg.split_ratio_test * n_total))
    n_val = int(np.round(cfg.split_ratio_val * n_total))
    n_train_full = n_total - n_test - n_val

    idx_train = indices[:n_train_full]
    idx_val = indices[n_train_full : n_train_full + n_val]
    idx_test = indices[n_train_full + n_val :]

    X_train_full = X_all[idx_train]
    y_train_full = y_all[idx_train]
    X_val = X_all[idx_val]
    y_val = y_all[idx_val]
    X_test = X_all[idx_test]
    y_test = y_all[idx_test]

    # --- Optional training-set subsampling ---
    if n_train_subsample is not None and n_train_subsample < len(X_train_full):
        sub_rng = np.random.default_rng(
            subsample_seed if subsample_seed is not None else cfg.subsample_seed
        )
        sub_idx = sub_rng.choice(len(X_train_full), size=n_train_subsample, replace=False)
        X_train = X_train_full[sub_idx]
        y_train_orig = y_train_full[sub_idx].copy()
    else:
        X_train = X_train_full
        y_train_orig = y_train_full.copy()

    # --- Fit scalers strictly on ACTIVE training data ---
    x_scaler = StandardScaler()
    x_scaler.fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    y_scaler: StandardScaler | None = None
    if cfg.standardize_y_for_training:
        y_scaler = StandardScaler()
        y_scaler.fit(y_train_orig)
        y_train_scaled = y_scaler.transform(y_train_orig)
        y_val = y_scaler.transform(y_val)
        y_test = y_scaler.transform(y_test)
    else:
        y_train_scaled = y_train_orig.copy()

    # --- Optional label randomization ---
    if randomize_train_labels:
        perm_rng = np.random.default_rng(
            label_perm_seed if label_perm_seed is not None else cfg.label_perm_seed
        )
        perm_idx = perm_rng.permutation(len(y_train_scaled))
        y_train = y_train_scaled[perm_idx].copy()
    else:
        y_train = y_train_scaled.copy()

    return RealDatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        y_train_original=y_train_orig,
        labels_randomized=randomize_train_labels,
        feature_names=feature_cols,
        target_name=target_col,
        target_unit="MPa",
        n_features=len(feature_cols),
        n_total=n_total,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        irreducible_mse=None,
    )


def load_concrete_kfold(
    n_splits: int = 5,
    cfg: RealDataConfig | None = None,
) -> list[RealDatasetSplit]:
    """Generate outer k-fold splits with strict per-fold train-only scaling.

    Each fold reserves 1/n_splits (20% = 206 samples) as an untouched test fold,
    splits the remaining 80% into 75% train (618) and 25% validation (206),
    and fits StandardScaler exclusively on that fold's training split.
    """
    cfg = cfg or RealDataConfig()
    paths = ProjectPaths()
    csv_path = Path(cfg.raw_csv_path)
    if not csv_path.exists():
        csv_path = paths.data_raw / "concrete.csv"

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "strength"]
    X_all = df[feature_cols].values.astype(np.float64)
    y_all = df[["strength"]].values.astype(np.float64)
    n_total = len(df)

    rng = np.random.default_rng(cfg.split_seed)
    perm = rng.permutation(n_total)
    test_folds = np.array_split(perm, n_splits)

    splits: list[RealDatasetSplit] = []
    for fold_idx in range(n_splits):
        idx_test = test_folds[fold_idx]
        idx_remain = np.concatenate([test_folds[j] for j in range(n_splits) if j != fold_idx])

        n_val = int(np.round(0.25 * len(idx_remain)))  # 25% of 80% = 20% overall
        idx_val = idx_remain[:n_val]
        idx_train = idx_remain[n_val:]

        X_tr, y_tr_orig = X_all[idx_train], y_all[idx_train]
        X_va, y_va = X_all[idx_val], y_all[idx_val]
        X_te, y_te = X_all[idx_test], y_all[idx_test]

        x_scaler = StandardScaler().fit(X_tr)
        X_tr_s = x_scaler.transform(X_tr)
        X_va_s = x_scaler.transform(X_va)
        X_te_s = x_scaler.transform(X_te)

        y_scaler: StandardScaler | None = None
        if cfg.standardize_y_for_training:
            y_scaler = StandardScaler().fit(y_tr_orig)
            y_tr_s = y_scaler.transform(y_tr_orig)
            y_va_s = y_scaler.transform(y_va)
            y_te_s = y_scaler.transform(y_te)
        else:
            y_tr_s, y_va_s, y_te_s = y_tr_orig.copy(), y_va.copy(), y_te.copy()

        splits.append(
            RealDatasetSplit(
                X_train=X_tr_s,
                y_train=y_tr_s,
                X_val=X_va_s,
                y_val=y_va_s,
                X_test=X_te_s,
                y_test=y_te_s,
                y_train_original=y_tr_orig.copy(),
                labels_randomized=False,
                feature_names=feature_cols,
                target_name="strength",
                target_unit="MPa",
                n_features=len(feature_cols),
                n_total=n_total,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                irreducible_mse=None,
            )
        )
    return splits


