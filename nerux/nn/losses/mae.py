import numpy as np
from ...tensor import Tensor




class MAELoss:
    def __call__(self, pred: Tensor, target: Tensor):
        return (pred - target).abs().mean()




