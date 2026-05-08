import numpy as np
from ...tensor import Tensor
from .base import Base


class GlobalAvgPool2D(Base):
    """

    Input:  (N, C, H, W)
    Output: (N, C, 1, 1) or (N, C) if keepdims=False

    Args:
        keepdims: If True, keeps spatial dimensions as 1x1. If False, removes them.
    """

    def __init__(self, keepdims=False):
        super().__init__()
        self.keepdims = keepdims

    def forward(self, x):
        result = x.mean(axis=(2, 3), keepdims=True)

        if not self.keepdims:
            batch_size = result.shape[0]
            num_channels = result.shape[1]
            result = result.reshape(batch_size, num_channels)

        return result
