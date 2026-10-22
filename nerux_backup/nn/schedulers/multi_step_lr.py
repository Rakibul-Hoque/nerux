from .scheduler import Scheduler


class MultiStep(Scheduler):
    """
    Decay lr by gamma at each milestone epoch.

    Example: MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)
    """

    def __init__(self, optimizer, milestones, gamma=0.1, **kwargs):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, **kwargs)

    def get_lr(self):
        n_passed = sum(1 for m in self.milestones if self.last_epoch >= m)
        return self._base_lr * (self.gamma**n_passed)

    def __repr__(self):
        return (
            f"MultiStepLR(base_lr={self._base_lr}, "
            f"milestones={self.milestones}, gamma={self.gamma})"
        )
