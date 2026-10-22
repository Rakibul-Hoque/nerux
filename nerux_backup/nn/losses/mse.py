from ...tensor import Tensor


class MSELoss:
    def __call__(self, pred: Tensor, target: Tensor):
        diff = pred - target
        return (diff * diff).mean()
