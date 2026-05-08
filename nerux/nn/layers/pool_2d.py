import numpy as np
from ...tensor import Tensor
from .base import Base
from ..functional.pool_2d import Pool2DFunction


class Pool2D(Base):


    def __init__(
        self,
        kernel_size,
        stride=None,
        padding="valid",
        padding_mode="constant",
        mode="max",
    ):
        super().__init__()

        # Normalize kernel_size to tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)

        # Normalize stride to tuple
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)

        self.padding = padding
        self.padding_mode = padding_mode

        if mode not in ["max", "avg"]:
            raise ValueError(f"mode must be 'max' or 'avg', got '{mode}'")
        self.mode = mode

    def forward(self, x):
        return Pool2DFunction.apply(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            padding_mode=self.padding_mode,
            mode=self.mode,
        )
