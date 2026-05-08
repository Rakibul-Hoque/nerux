from ...tensor import Tensor

class BCELoss:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, pred: Tensor, target: Tensor):
        pred = pred.clip(self.eps, 1 - self.eps)
        loss = -(target * pred.log() + (1 - target) * (1 - pred).log()).mean()
        return loss
