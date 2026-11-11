"""Unit and integration verification tests for Project 9.

Tests mathematical correctness, gradient verification, leakage prevention,
reproducibility, and early-stopping checkpoint restoration.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.experiment_config import RealDataConfig, SyntheticDataConfig, TrainConfig
from src.data.real_data import load_concrete_dataset
from src.data.synthetic_data import build_synthetic_dataset
from src.experiments.gradient_check import run_gradient_check
from src.models.linear_baseline import LinearBaseline
from src.models.neural_network_numpy import NeuralNetworkNumPy
from src.training.trainer import train_model


class TestProject9Pipeline(unittest.TestCase):
    def test_parameter_counts(self) -> None:
        mlp = NeuralNetworkNumPy(input_dim=8, hidden_width=16, activation="tanh", seed=0)
        self.assertEqual(mlp.count_parameters(), 16 * (8 + 2) + 1)

        lin = LinearBaseline()
        X = np.random.default_rng(0).normal(size=(20, 8))
        y = np.random.default_rng(1).normal(size=(20, 1))
        lin.fit(X, y)
        self.assertEqual(lin.count_parameters(), 9)

    def test_gradient_verification_tanh_and_relu(self) -> None:
        for act in ("tanh", "relu"):
            for lam in (0.0, 0.01):
                res = run_gradient_check(
                    activation=act,
                    l2_lambda=lam,
                    verbose=False,
                )
                self.assertTrue(res["overall_pass_fd"], f"FD failed for {act}, lam={lam}")
                self.assertTrue(res["overall_pass_pt"], f"PT failed for {act}, lam={lam}")

    def test_synthetic_split_invariance_and_label_randomization(self) -> None:
        cfg = SyntheticDataConfig()
        ds_norm = build_synthetic_dataset(cfg=cfg, n_train=100, randomize_train_labels=False)
        ds_rand = build_synthetic_dataset(cfg=cfg, n_train=100, randomize_train_labels=True)
        ds_big = build_synthetic_dataset(cfg=cfg, n_train=400, randomize_train_labels=False)

        # Val and test sets must remain bitwise identical across train sizes and label randomization
        np.testing.assert_array_equal(ds_norm.X_val, ds_rand.X_val)
        np.testing.assert_array_equal(ds_norm.y_val, ds_rand.y_val)
        np.testing.assert_array_equal(ds_norm.X_test, ds_rand.X_test)
        np.testing.assert_array_equal(ds_norm.y_test, ds_rand.y_test)
        np.testing.assert_array_equal(ds_norm.X_val, ds_big.X_val)
        np.testing.assert_array_equal(ds_norm.y_val, ds_big.y_val)

        # Randomized training labels preserve input features and target multiset, break alignment
        np.testing.assert_array_equal(ds_norm.X_train, ds_rand.X_train)
        self.assertFalse(np.array_equal(ds_norm.y_train, ds_rand.y_train))
        np.testing.assert_allclose(np.sort(ds_norm.y_train.ravel()), np.sort(ds_rand.y_train.ravel()))

    def test_concrete_no_data_leakage_and_scaler_isolation(self) -> None:
        cfg = RealDataConfig()
        rd_full = load_concrete_dataset(cfg=cfg, randomize_train_labels=False)
        rd_rand = load_concrete_dataset(cfg=cfg, randomize_train_labels=True)
        rd_sub = load_concrete_dataset(cfg=cfg, n_train_subsample=50)

        # Scaler must depend ONLY on active training subset
        self.assertFalse(np.allclose(rd_full.x_scaler.mean_, rd_sub.x_scaler.mean_))
        # Train mean must be ~0 and std ~1 on scaled X_train
        np.testing.assert_allclose(rd_full.X_train.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(rd_full.X_train.std(axis=0), 1.0, atol=1e-12)
        # Val/test targets in original units must never be permuted when randomize_train_labels=True
        np.testing.assert_allclose(
            rd_full.inverse_transform_y(rd_full.y_val),
            rd_rand.inverse_transform_y(rd_rand.y_val),
        )
        np.testing.assert_allclose(
            rd_full.inverse_transform_y(rd_full.y_test),
            rd_rand.inverse_transform_y(rd_rand.y_test),
        )
        # Total split coverage = 1030
        self.assertEqual(
            len(rd_full.X_train) + len(rd_full.X_val) + len(rd_full.X_test),
            1030,
        )

    def test_seed_determinism(self) -> None:
        ds = build_synthetic_dataset(n_train=40)
        cfg = TrainConfig(optimizer="adam", learning_rate=0.01, max_epochs=100)
        m1 = NeuralNetworkNumPy(input_dim=1, hidden_width=8, activation="tanh", seed=123)
        m2 = NeuralNetworkNumPy(input_dim=1, hidden_width=8, activation="tanh", seed=123)
        h1 = train_model(m1, ds.X_train, ds.y_train, ds.X_val, ds.y_val, cfg)
        h2 = train_model(m2, ds.X_train, ds.y_train, ds.X_val, ds.y_val, cfg)
        np.testing.assert_array_equal(h1.train_mse, h2.train_mse)
        np.testing.assert_array_equal(m1.predict(ds.X_test), m2.predict(ds.X_test))

    def test_early_stopping_restores_best_checkpoint(self) -> None:
        ds = build_synthetic_dataset(n_train=30)
        m = NeuralNetworkNumPy(input_dim=1, hidden_width=64, activation="tanh", seed=42)
        cfg = TrainConfig(
            optimizer="adam",
            learning_rate=0.02,
            max_epochs=600,
            early_stopping=True,
            patience=20,
            min_delta=1e-5,
            restore_best=True,
        )
        hist = train_model(m, ds.X_train, ds.y_train, ds.X_val, ds.y_val, cfg)
        if hist.early_stopped and hist.best_val_epoch is not None:
            restored_val_mse = float(np.mean((m.predict(ds.X_val) - ds.y_val) ** 2))
            self.assertAlmostEqual(restored_val_mse, hist.val_mse[hist.best_val_epoch], places=10)

    def test_pytorch_training_trajectory_match(self) -> None:
        import torch
        import torch.nn as nn
        from src.models.neural_network_torch import NeuralNetworkTorch

        ds = build_synthetic_dataset(n_train=30)
        np_mlp = NeuralNetworkNumPy(input_dim=1, hidden_width=8, activation="tanh", seed=99)
        pt_mlp = NeuralNetworkTorch(input_dim=1, hidden_width=8, activation="tanh")
        pt_mlp.copy_params_from_numpy(np_mlp)

        # Verify identical initial forward predictions
        np.testing.assert_allclose(np_mlp.predict(ds.X_val), pt_mlp.predict(ds.X_val), rtol=1e-12, atol=1e-12)

        # Train both for 50 Adam steps with identical hyperparameters
        cfg = TrainConfig(optimizer="adam", learning_rate=0.01, max_epochs=50)
        hist = train_model(np_mlp, ds.X_train, ds.y_train, ds.X_val, ds.y_val, cfg)

        opt = torch.optim.Adam(pt_mlp.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        loss_fn = nn.MSELoss(reduction="mean")
        Xt = torch.from_numpy(ds.X_train).double()
        Yt = torch.from_numpy(ds.y_train).double()
        for _ in range(50):
            opt.zero_grad()
            loss = loss_fn(pt_mlp(Xt), Yt)
            loss.backward()
            opt.step()

        np.testing.assert_allclose(np_mlp.predict(ds.X_val), pt_mlp.predict(ds.X_val), rtol=1e-8, atol=1e-8)

    def test_early_stopping_warmup_prevents_early_halt(self) -> None:
        from src.training.early_stopping import EarlyStopping

        es = EarlyStopping(patience=2, min_delta=1e-3, warmup_epochs=5, restore_best=False)
        # Flat plateau during warmup epochs 0..4 must not trigger stop
        for ep in range(5):
            self.assertFalse(es.check(ep, 1.0))
        # Patience starts counting at epoch >= 5
        self.assertFalse(es.check(5, 1.0))  # wait=1
        self.assertTrue(es.check(6, 1.0))   # wait=2 -> stop

    def test_concrete_kfold_no_leakage_and_coverage(self) -> None:
        from src.data.real_data import load_concrete_kfold

        folds = load_concrete_kfold(n_splits=5)
        self.assertEqual(len(folds), 5)
        test_means = [float(f.inverse_transform_y(f.y_test).mean()) for f in folds]
        # Distinct test folds must exist and total split size per fold == 1030
        self.assertEqual(len(set(round(m, 6) for m in test_means)), 5)
        for f in folds:
            self.assertEqual(len(f.X_train) + len(f.X_val) + len(f.X_test), 1030)
            np.testing.assert_allclose(f.X_train.mean(axis=0), 0.0, atol=1e-12)
            np.testing.assert_allclose(f.X_train.std(axis=0), 1.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
