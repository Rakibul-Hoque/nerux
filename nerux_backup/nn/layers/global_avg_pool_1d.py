import numpy as np
from .base import Base


class GlobalAvgPool1D(Base):
    """
    Input:  (N, C, L)
    Output: (N, C, 1) or (N, C) if keepdims=False

    Args:
        keepdims: If True, keeps length dimension as 1. If False, removes it.
    """

    def __init__(self, keepdims=False):
        super().__init__()
        self.keepdims = keepdims

    def forward(self, x):
        result = x.mean(axis=2, keepdims=True)
        if not self.keepdims:
            batch_size = result.shape[0]
            num_channels = result.shape[1]
            result = result.reshape(batch_size, num_channels)

        return result
