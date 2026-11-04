from ..parameter import Parameter
from ...tensor import factory as init
from .base import Base
from ..functional._nd import normalize_tuple
from ..functional.conv1d import Conv1DFunction
import numpy as np


class Conv1D(Base):
    """
    Input:  (N, C_in, L)
    Output: (N, C_out, ...)
    """

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
        self.kernel_size = normalize_tuple(kernel_size, 1, "kernel_size")
        self.stride = normalize_tuple(stride, 1, "stride")
        self.padding = padding
        self.padding_mode = padding_mode
        self.use_bias = bias

    def build(self, in_shape):
        if len(in_shape) != 1 + 1:
            raise ValueError(
                f"Conv1D expects input shape (C, L), got {tuple(in_shape)}"
            )
        C_in = in_shape[0]
        scale = np.sqrt(2.0 / (C_in * int(np.prod(self.kernel_size))))
        W = np.random.randn(self.out_channels, C_in, *self.kernel_size) * scale
        self.W = self.add_parameter("W", Parameter(W))
        if self.use_bias:
            self.B = self.add_parameter("B", Parameter(np.zeros((self.out_channels,))))
        else:
            self.B = None

    def forward(self, x):
        b = self.B if self.use_bias else init.zeros((self.out_channels,))
        return Conv1DFunction.apply(
            x,
            self.W,
            b,
            stride=self.stride,
            padding=self.padding,
            padding_mode=self.padding_mode,
        )
