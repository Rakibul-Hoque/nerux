from .scheduler import Scheduler
import numpy as np


class Linear(Scheduler):
    """
    Linearly anneal lr from start_factor*base_lr to end_factor*base_lr
    over total_iters epochs, then hold.

    Example: LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=30)
    """

    def __init__(
        self, optimizer, start_factor=1.0, end_factor=0.0, total_epochs=100, **kwargs
    ):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_epochs = total_epochs
        super().__init__(optimizer, **kwargs)

    def get_lr(self):
        t = min(self.last_epoch, self.total_epochs)
        factor = self.start_factor + (self.end_factor - self.start_factor) * (
            t / self.total_epochs
        )
        return self._base_lr * factor

    def __repr__(self):
        return (
            f"LinearLR(base_lr={self._base_lr}, "
            f"start_factor={self.start_factor}, end_factor={self.end_factor}, "
            f"total_epochs={self.total_epochs})"
        )
