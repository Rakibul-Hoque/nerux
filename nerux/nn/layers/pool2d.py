from .base import Base
from ..functional._nd import normalize_tuple
from ..functional.pool_2d import Pool2DFunction


class Pool2D(Base):
    """Max / average pooling over (N, C, H, W)."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding="valid",
        padding_mode="constant",
        mode="max",
    ):
        super().__init__()
        self.kernel_size = normalize_tuple(kernel_size, 2, "kernel_size")
        self.stride = (
            self.kernel_size
            if stride is None
            else normalize_tuple(stride, 2, "stride")
        )
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
