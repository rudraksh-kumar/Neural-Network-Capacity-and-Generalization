"""Unified training loop for the NumPy MLP with full epoch logging.

Supports:
- Full-batch and mini-batch training.
- Coupled L2 regularisation (via model.backward) and decoupled weight decay (via AdamW).
- Early stopping with best-checkpoint restoration.
- Computational budget limits (max_epochs, max_steps, max_flops).
- Convergence detection (Protocol B).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.models.neural_network_numpy import NeuralNetworkNumPy
from src.training.optimizers import Optimizer, build_optimizer
from src.training.early_stopping import EarlyStopping
from config.experiment_config import TrainConfig


@dataclass
class TrainHistory:
    """Per-epoch training log."""

    train_mse: list[float] = field(default_factory=list)
    val_mse: list[float] = field(default_factory=list)
    train_reg_loss: list[float] = field(default_factory=list)  # MSE + L2 penalty
    epochs_run: int = 0
    steps_run: int = 0
    total_flops: float = 0.0
    wall_time_sec: float = 0.0
    converged: bool = False
    early_stopped: bool = False
    stopped_epoch: int | None = None
    best_val_epoch: int | None = None
    budget_exhausted: bool = False


def train_model(
    model: NeuralNetworkNumPy,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig | None = None,
    optimizer: Optimizer | None = None,
    verbose: bool = False,
    log_every: int = 100,
) -> TrainHistory:
    """Train *model* in-place and return epoch-level history.

    Parameters
    ----------
    model : NeuralNetworkNumPy
        The MLP to train (modified in-place).
    X_train, y_train : ndarray
        Training data (already scaled/preprocessed).
    X_val, y_val : ndarray
        Validation data (same preprocessing).
    cfg : TrainConfig
        Optimiser, regularisation, early-stopping, and budget parameters.
    optimizer : Optimizer | None
        Pre-built optimizer; if None, one is created from *cfg*.
    verbose : bool
        Print progress every *log_every* epochs.
    log_every : int
        Epoch interval for verbose printing.
    """
    cfg = cfg or TrainConfig()

    # Ensure arrays are float64 and 2D
    X_train = np.asarray(X_train, dtype=np.float64).reshape(-1, model.input_dim)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    X_val = np.asarray(X_val, dtype=np.float64).reshape(-1, model.input_dim)
    y_val = np.asarray(y_val, dtype=np.float64).reshape(-1, 1)

    B_train = X_train.shape[0]
    batch_size = cfg.batch_size if cfg.batch_size is not None else B_train
    flops_per_step = model.flops_per_step(batch_size)

    # Build optimizer
    if optimizer is None:
        opt_kwargs = dict(
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            regularize_biases=cfg.regularize_biases,
        )
        opt = build_optimizer(cfg.optimizer, **opt_kwargs)
    else:
        opt = optimizer

    # Early stopping
    es: EarlyStopping | None = None
    if cfg.early_stopping:
        es = EarlyStopping(
            patience=cfg.patience,
            min_delta=cfg.min_delta,
            restore_best=cfg.restore_best,
            warmup_epochs=cfg.warmup_epochs,
        )

    # Convergence detection (Protocol B)
    conv_tol = cfg.convergence_tol
    conv_patience = cfg.convergence_patience
    conv_wait = 0
    prev_val_loss = float("inf")

    history = TrainHistory()
    rng = np.random.default_rng(0)
    t_start = time.perf_counter()

    # Force per-epoch validation evaluation when early stopping or convergence stopping is active
    eval_every = 1 if (es is not None or conv_tol is not None) else max(1, int(cfg.eval_val_every))
    last_val_mse = float("nan")

    for epoch in range(cfg.max_epochs):
        # --- Mini-batch shuffling ---
        if cfg.shuffle_batches and batch_size < B_train:
            perm = rng.permutation(B_train)
            X_train_epoch = X_train[perm]
            y_train_epoch = y_train[perm]
        else:
            X_train_epoch = X_train
            y_train_epoch = y_train

        # --- Training steps ---
        n_batches = max(1, int(np.ceil(B_train / batch_size)))
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, B_train)
            Xb = X_train_epoch[start:end]
            yb = y_train_epoch[start:end]

            # Forward
            Y_hat = model.forward(Xb)

            # Backward (coupled L2 if using Adam; grads exclude L2 if using AdamW)
            l2_for_backward = cfg.l2_lambda if cfg.optimizer.lower() != "adamw" else 0.0
            grads = model.backward(
                yb,
                l2_lambda=l2_for_backward,
                regularize_biases=cfg.regularize_biases,
            )

            # Optimizer step
            params = model.get_params()
            opt.step(params, grads)
            # params are views, so model is updated in-place

            history.steps_run += 1
            history.total_flops += flops_per_step

            # Step budget
            if cfg.max_steps is not None and history.steps_run >= cfg.max_steps:
                history.budget_exhausted = True
                break
            if cfg.max_flops is not None and history.total_flops >= cfg.max_flops:
                history.budget_exhausted = True
                break

        # --- Epoch-level evaluation ---
        is_last = (epoch == cfg.max_epochs - 1) or history.budget_exhausted
        should_eval_val = is_last or (epoch == 0) or ((epoch + 1) % eval_every == 0)

        y_hat_train = model.predict(X_train)
        train_mse = model.mse_loss(y_hat_train, y_train)
        train_reg = model.mse_loss_with_l2(
            y_hat_train, y_train, model.W1, model.W2, cfg.l2_lambda
        )

        if should_eval_val:
            y_hat_val = model.predict(X_val)
            last_val_mse = model.mse_loss(y_hat_val, y_val)
        val_mse = last_val_mse

        history.train_mse.append(train_mse)
        history.val_mse.append(val_mse)
        history.train_reg_loss.append(train_reg)
        history.epochs_run = epoch + 1

        if verbose and (epoch % log_every == 0 or epoch == cfg.max_epochs - 1):
            print(f"  Epoch {epoch:5d} | train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

        # --- Early stopping ---
        if es is not None:
            should_stop = es.check(epoch, val_mse, model.get_params())
            if should_stop:
                history.early_stopped = True
                history.stopped_epoch = epoch
                history.best_val_epoch = es.best_epoch
                if es.restore_best and es.best_params is not None:
                    model.set_params(es.best_params)
                break

        # --- Convergence detection (Protocol B) ---
        if conv_tol is not None:
            improvement = prev_val_loss - val_mse
            if improvement < conv_tol:
                conv_wait += 1
            else:
                conv_wait = 0
            prev_val_loss = val_mse
            if conv_wait >= conv_patience:
                history.converged = True
                break

        # --- Budget exhaustion ---
        if history.budget_exhausted:
            break

    history.wall_time_sec = time.perf_counter() - t_start

    # Track best validation epoch if early stopping was not used
    if history.best_val_epoch is None and history.val_mse:
        history.best_val_epoch = int(np.argmin(history.val_mse))

    return history

