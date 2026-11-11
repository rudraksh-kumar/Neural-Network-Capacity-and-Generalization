"""Pilot validation: data pipelines, linear baseline, and MLP training on both datasets."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data.synthetic_data import build_synthetic_dataset
from src.data.real_data import load_concrete_dataset
from src.models.linear_baseline import LinearBaseline
from src.models.neural_network_numpy import NeuralNetworkNumPy
from src.training.trainer import train_model
from src.evaluation.metrics import compute_regression_metrics
from config.experiment_config import TrainConfig

# --- Synthetic dataset ---
ds = build_synthetic_dataset()
print("=== Synthetic Dataset ===")
print(f"Train: X={ds.X_train.shape}, y={ds.y_train.shape}")
print(f"Val:   X={ds.X_val.shape}, y={ds.y_val.shape}")
print(f"Test:  X={ds.X_test.shape}, y={ds.y_test.shape}")
print(f"Irreducible MSE (Bayes): {ds.irreducible_mse:.4f}")

# --- Concrete dataset ---
rd = load_concrete_dataset()
print(f"\n=== Concrete Dataset ===")
print(f"Train: X={rd.X_train.shape}, y={rd.y_train.shape}")
print(f"Val:   X={rd.X_val.shape}, y={rd.y_val.shape}")
print(f"Test:  X={rd.X_test.shape}, y={rd.y_test.shape}")
print(f"Features: {rd.feature_names}")

# --- Linear baseline (synthetic) ---
lin = LinearBaseline()
lin.fit(ds.X_train, ds.y_train)
y_pred_lin = lin.predict(ds.X_test)
m_lin = compute_regression_metrics(ds.y_test, y_pred_lin)
print(f"\n=== Linear Baseline (Synthetic Test) ===")
print(f"MSE={m_lin['mse']:.4f}  RMSE={m_lin['rmse']:.4f}  R2={m_lin['r2']:.4f}")

# --- MLP pilot (synthetic) ---
model = NeuralNetworkNumPy(input_dim=1, hidden_width=32, activation="tanh", seed=42)
cfg = TrainConfig(optimizer="adam", learning_rate=0.01, max_epochs=1500)
hist = train_model(model, ds.X_train, ds.y_train, ds.X_val, ds.y_val, cfg=cfg)
y_pred_nn = model.predict(ds.X_test)
m_nn = compute_regression_metrics(ds.y_test, y_pred_nn)
print(f"\n=== MLP H=32 tanh (Synthetic Test) ===")
print(f"Epochs: {hist.epochs_run}, Steps: {hist.steps_run}")
print(f"Final train MSE={hist.train_mse[-1]:.4f}, val MSE={hist.val_mse[-1]:.4f}")
print(f"Test: MSE={m_nn['mse']:.4f}  RMSE={m_nn['rmse']:.4f}  R2={m_nn['r2']:.4f}")
print(f"Bayes floor={ds.irreducible_mse:.4f}  Gap to floor={m_nn['mse'] - ds.irreducible_mse:.4f}")

# --- Linear baseline (concrete) ---
lin_r = LinearBaseline()
lin_r.fit(rd.X_train, rd.y_train)
y_pred_lr = lin_r.predict(rd.X_test)
m_lr = compute_regression_metrics(rd.y_test, y_pred_lr)
print(f"\n=== Linear Baseline (Concrete Test) ===")
print(f"MSE={m_lr['mse']:.4f}  RMSE={m_lr['rmse']:.4f}  R2={m_lr['r2']:.4f}")

# --- MLP pilot (concrete) ---
model_r = NeuralNetworkNumPy(input_dim=8, hidden_width=32, activation="tanh", seed=42)
cfg_r = TrainConfig(optimizer="adam", learning_rate=0.005, max_epochs=2000)
hist_r = train_model(model_r, rd.X_train, rd.y_train, rd.X_val, rd.y_val, cfg=cfg_r)
y_pred_nnr = model_r.predict(rd.X_test)
m_nnr = compute_regression_metrics(rd.y_test, y_pred_nnr)
print(f"\n=== MLP H=32 tanh (Concrete Test) ===")
print(f"Epochs: {hist_r.epochs_run}")
print(f"Final train MSE={hist_r.train_mse[-1]:.4f}, val MSE={hist_r.val_mse[-1]:.4f}")
print(f"Test: MSE={m_nnr['mse']:.4f}  RMSE={m_nnr['rmse']:.4f}  R2={m_nnr['r2']:.4f}")
