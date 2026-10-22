import numpy as np
from .optimizer import Optimizer


class RMSProp(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        beta=0.9,
        eps=1e-8,
        grad_clip=None,
        grad_clip_mode="norm",
    ):
        super().__init__(params, lr,grad_clip,grad_clip_mode)
        self.beta, self.eps = beta, eps
        self.avg_sq_grad = [np.zeros_like(p.data) for p in self.params]

    def _step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self.avg_sq_grad[i] = self.beta * self.avg_sq_grad[i] + (1 - self.beta) * (
                g * g
            )
            p.data -= self.lr * g / (np.sqrt(self.avg_sq_grad[i]) + self.eps)
