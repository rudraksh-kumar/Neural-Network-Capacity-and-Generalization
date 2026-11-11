"""Early stopping with patience, minimum-improvement threshold, and best-checkpoint restoration."""

from __future__ import annotations

import numpy as np


class EarlyStopping:
    """Track validation metric and signal when training should stop.

    Parameters
    ----------
    patience : int
        Number of epochs without improvement before stopping.
    min_delta : float
        Minimum decrease in validation loss to qualify as improvement.
    restore_best : bool
        If True, ``best_params`` are saved at every improvement.
    """

    def __init__(
        self,
        patience: int = 60,
        min_delta: float = 1e-5,
        restore_best: bool = True,
        warmup_epochs: int = 0,
    ) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best = restore_best
        self.warmup_epochs = max(0, int(warmup_epochs))

        self.best_loss: float = float("inf")
        self.best_epoch: int = 0
        self.wait: int = 0
        self.stopped: bool = False
        self.best_params: dict[str, np.ndarray] | None = None

    def check(
        self,
        epoch: int,
        val_loss: float,
        params: dict[str, np.ndarray] | None = None,
    ) -> bool:
        """Return True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best and params is not None:
                self.best_params = {k: v.copy() for k, v in params.items()}
            return False

        # During warmup, track best loss/checkpoint but do not accumulate patience
        if epoch < self.warmup_epochs:
            self.wait = 0
            return False

        self.wait += 1
        if self.wait >= self.patience:
            self.stopped = True
            return True
        return False

    def reset(self) -> None:
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.wait = 0
        self.stopped = False
        self.best_params = None

