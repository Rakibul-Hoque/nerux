from .scheduler import Scheduler
import numpy as np


class ReduceLROnPlateau:
    """
    Monitor a metric and reduce lr by factor when it stops improving.
    Not a LRScheduler subclass — call scheduler.step(metric) instead of step().

    Example:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        scheduler.step(val_loss)
    """

    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        min_lr=0.0,
        threshold=1e-4,
        last_epoch=-1,
        verbose=False,
    ):
        assert mode in ("min", "max"), "mode must be 'min' or 'max'"
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.last_epoch = last_epoch
        self.verbose = verbose
        self._base_lr = optimizer.lr
        self._best = np.inf if mode == "min" else -np.inf
        self._wait = 0
        self._history = []

    def step(self, metric):
        self.last_epoch += 1
        improved = (
            metric < self._best - self.threshold
            if self.mode == "min"
            else metric > self._best + self.threshold
        )
        if improved:
            self._best = metric
            self._wait = 0
            if self.verbose:
                print(
                    f"[Scheduler] epoch {self.last_epoch}: lr = {self.optimizer.lr:.8f}"
                )
        else:
            self._wait += 1
            if self._wait >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                if self.verbose:
                    print(
                        f"[ReduceLROnPlateau] lr {self.optimizer.lr:.6f} → {new_lr:.6f} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                self.optimizer.lr = new_lr
                self._wait = 0
        self._history.append((metric, self.optimizer.lr))

    def get_last_lr(self):
        return self._history[-1][1] if self._history else self._base_lr

    def get_history(self):
        return list(self._history)

    def __repr__(self):
        return (
            f"ReduceLROnPlateau(mode={self.mode}, factor={self.factor}, "
            f"patience={self.patience}, min_lr={self.min_lr})"
        )
