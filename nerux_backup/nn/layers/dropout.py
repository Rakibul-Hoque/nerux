import numpy as np
from ...tensor import Tensor
from .base import Base


class Dropout(Base):
    """
    Dropout layer for regularization.
    During training, randomly sets a fraction of inputs to 0.
    During inference, passes inputs through unchanged.
    """

    def __init__(self, p=0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in [0, 1)")
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training or self.p == 0:
            return x


        keep_prob = 1 - self.p
        self.mask = (np.random.rand(*x.shape) < keep_prob).astype(float)


        return x * Tensor(self.mask / keep_prob)

