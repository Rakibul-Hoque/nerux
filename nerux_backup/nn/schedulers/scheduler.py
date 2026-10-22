import numpy as np


class Scheduler:
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose
        self._base_lr = optimizer.lr  # original lr — never mutated
        self._history = []  # (epoch, lr) log

        # Apply initial lr
        self.step()

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        new_lr = self.get_lr()
        self.optimizer.lr = new_lr
        self._history.append((self.last_epoch, new_lr))
        if self.verbose:
            print(f"[Scheduler] epoch {self.last_epoch}: lr = {new_lr:.8f}")

    def get_last_lr(self):
        return self._history[-1][1] if self._history else self._base_lr

    def get_history(self):
        return list(self._history)

    def reset(self):

        self.last_epoch = -1
        self._history.clear()
        self.optimizer.lr = self._base_lr
        self.step()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"base_lr={self._base_lr}, "
            f"last_epoch={self.last_epoch})"
        )
