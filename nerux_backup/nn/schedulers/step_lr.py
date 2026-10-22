from .scheduler import Scheduler


class Step(Scheduler):
    """
    Decay lr by gamma every step_size epochs.

    Example: StepLR(optimizer, step_size=10, gamma=0.5)
    epoch  0-9:  lr = base_lr
    epoch 10-19: lr = base_lr * 0.5
    epoch 20-29: lr = base_lr * 0.25
    """

    def __init__(self, optimizer, step_size, gamma=0.1, **kwargs):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, **kwargs)

    def get_lr(self):
        if self.last_epoch == 0:
            return self._base_lr
        return self._base_lr * (self.gamma ** (self.last_epoch // self.step_size))

    def __repr__(self):
        return (
            f"StepLR(base_lr={self._base_lr}, "
            f"step_size={self.step_size}, gamma={self.gamma})"
        )
