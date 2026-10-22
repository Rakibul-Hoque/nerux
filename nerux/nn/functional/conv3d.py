from .function import Function
from ._nd import conv_forward_nd, conv_backward_nd


class Conv3DFunction(Function):
    """
    VECTORIZED 3D convolution via im2col.
    x: (N, C_in, D, H, W)   w: (C_out, C_in, kernel...)   b: (C_out,)
    """

    def forward(self, x, w, b, stride=1, padding="valid", padding_mode="constant"):
        out, self.cache = conv_forward_nd(
            x, w, b, stride, padding, padding_mode, ndim=3
        )
        return out

    def backward(self, grad_output):
        return conv_backward_nd(grad_output, self.cache)
