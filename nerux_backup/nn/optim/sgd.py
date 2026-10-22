from .optimizer import Optimizer
import numpy as np


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0,grad_clip=None,
        grad_clip_mode="norm",):
        super().__init__(params, lr,grad_clip,grad_clip_mode)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def _step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * g
            p.data += self.velocities[i]
