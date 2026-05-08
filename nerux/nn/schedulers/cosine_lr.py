from .scheduler import Scheduler
import numpy as np


class CosineAnnealing(Scheduler):
    """
    Cosine annealing between base_lr and min_lr over T_max epochs,
    then repeats.

    Example: CosineAnnealingLR(optimizer, T_max=50, min_lr=1e-6)
    """

    def __init__(self, optimizer, T_max, min_lr=0.0, **kwargs):
        self.T_max = T_max
        self.min_lr = min_lr
        super().__init__(optimizer, **kwargs)

    def get_lr(self):
        t = self.last_epoch % self.T_max
        cos = np.cos(np.pi * t / self.T_max)
        return self.min_lr + 0.5 * (self._base_lr - self.min_lr) * (1 + cos)

    def __repr__(self):
        return (
            f"CosineAnnealingLR(base_lr={self._base_lr}, "
            f"T_max={self.T_max}, min_lr={self.min_lr})"
        )
