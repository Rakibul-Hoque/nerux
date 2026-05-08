import numpy as np
from ...tensor import Tensor
from .base import Base


class Flatten(Base):
    """
    Flattens input while preserving batch dimension.
    Input: (N, C, H, W) or any shape
    Output: (N, C*H*W*...)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
