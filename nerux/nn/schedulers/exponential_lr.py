from .scheduler import Scheduler


class Exponential(Scheduler):
    """
    Decay lr by gamma every epoch: lr = base_lr * gamma^epoch.

    Example: ExponentialLR(optimizer, gamma=0.95)
    """

    def __init__(self, optimizer, gamma, **kwargs):
        self.gamma = gamma
        super().__init__(optimizer, **kwargs)

    def get_lr(self):
        return self._base_lr * (self.gamma**self.last_epoch)

    def __repr__(self):
        return f"ExponentialLR(base_lr={self._base_lr}, gamma={self.gamma})"
