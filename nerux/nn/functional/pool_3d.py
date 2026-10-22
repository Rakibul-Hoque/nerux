from .function import Function
from ._nd import pool_forward_nd, pool_backward_nd


class Pool3DFunction(Function):
    """VECTORIZED 3D pooling (max / avg) via im2col. x: (N, C, D, H, W)"""

    def forward(
        self,
        x,
        kernel_size=2,
        stride=None,
        padding="valid",
        padding_mode="constant",
        mode="max",
    ):
        out, self.cache = pool_forward_nd(
            x, kernel_size, stride, padding, padding_mode, mode, ndim=3
        )
        return out

    def backward(self, grad_output):
        return pool_backward_nd(grad_output, self.cache)
