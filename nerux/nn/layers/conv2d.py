from ...tensor import factory as init
from .base import Base
from ..functional.conv2d import Conv2DFunction
import numpy as np


class Conv2D(Base):
    def __init__(
        self,
        out_channels,
        kernel_size,
        stride=1,
        padding="valid",
        padding_mode="constant",
        bias=True,
    ):
        super().__init__()
        self.out_channels = out_channels

      
        self.kernel_size = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = padding
        self.padding_mode = padding_mode
        self.use_bias = bias

    def build(self, in_shape):
        C_in = in_shape[0]
        KH, KW = self.kernel_size
        scale = np.sqrt(2.0 / (C_in * KH * KW))
        W = np.random.randn(self.out_channels, C_in, KH, KW) * scale
        self.W = self.add_parameter("W", init.tensor(W, requires_grad=True))
        if self.use_bias:
            self.B = self.add_parameter(
                "B", init.tensor(np.zeros((self.out_channels,)), requires_grad=True)
            )
        else:
            self.B = None

    def forward(self, x):
        b = self.B if self.use_bias else init.zeros((self.out_channels,))
        return Conv2DFunction.apply(
            x,
            self.W,
            b,
            stride=self.stride,
            padding=self.padding,
            padding_mode=self.padding_mode,
        )
