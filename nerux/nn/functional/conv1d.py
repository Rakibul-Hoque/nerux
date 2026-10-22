from .function import Function
from ._nd import conv_forward_nd, conv_backward_nd


class Conv1DFunction(Function):
    """
    VECTORIZED 1D convolution via im2col.
    x: (N, C_in, L)   w: (C_out, C_in, kernel...)   b: (C_out,)
    """

    def forward(self, x, w, b, stride=1, padding="valid", padding_mode="constant"):
        out, self.cache = conv_forward_nd(
            x, w, b, stride, padding, padding_mode, ndim=1
        )
        return out

    def backward(self, grad_output):
        return conv_backward_nd(grad_output, self.cache)
