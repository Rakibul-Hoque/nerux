from ...tensor import factory as init
from .base import Base
from ..functional._nd import normalize_tuple
from ..functional.conv3d import Conv3DFunction
import numpy as np


class Conv3D(Base):
    """
    Input:  (N, C_in, D, H, W)
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
        self.kernel_size = normalize_tuple(kernel_size, 3, "kernel_size")
        self.stride = normalize_tuple(stride, 3, "stride")
        self.padding = padding
        self.padding_mode = padding_mode
        self.use_bias = bias

    def build(self, in_shape):
        if len(in_shape) != 3 + 1:
            raise ValueError(
                f"Conv3D expects input shape (C, D, H, W), got {tuple(in_shape)}"
            )
        C_in = in_shape[0]
        scale = np.sqrt(2.0 / (C_in * int(np.prod(self.kernel_size))))
        W = np.random.randn(self.out_channels, C_in, *self.kernel_size) * scale
        self.W = self.add_parameter("W", init.tensor(W, requires_grad=True))
        if self.use_bias:
            self.B = self.add_parameter(
                "B", init.tensor(np.zeros((self.out_channels,)), requires_grad=True)
            )
        else:
            self.B = None

    def forward(self, x):
        b = self.B if self.use_bias else init.zeros((self.out_channels,))
        return Conv3DFunction.apply(
            x,
            self.W,
            b,
            stride=self.stride,
            padding=self.padding,
            padding_mode=self.padding_mode,
        )
