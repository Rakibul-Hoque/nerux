import numpy as np
from .function import Function


def calculate_same_padding(input_size, kernel_size, stride):
  
    output_size = (input_size + stride - 1) // stride
    padding_needed = max(0, (output_size - 1) * stride + kernel_size - input_size)
    pad_before = padding_needed // 2
    pad_after = padding_needed - pad_before
    return pad_before, pad_after


def im2col(
    x, kernel_h, kernel_w, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right
):
    """
    Efficient im2col implementation for convolution.
    Transforms image into column matrix for fast convolution.

    Args:
        x: (N, C, H, W)
        kernel_h, kernel_w: kernel dimensions
        stride_h, stride_w: stride values
        pad_*: padding values

    Returns:
        col: (N * H_out * W_out, C * KH * KW)
        output shape: (H_out, W_out)
    """
    N, C, H, W = x.shape

    # Apply padding
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
        )
    else:
        x_padded = x

    H_padded, W_padded = x_padded.shape[2], x_padded.shape[3]
    H_out = (H_padded - kernel_h) // stride_h + 1
    W_out = (W_padded - kernel_w) // stride_w + 1

    # Create column matrix
    col = np.zeros((N, C, kernel_h, kernel_w, H_out, W_out))

    for y in range(kernel_h):
        y_max = y + stride_h * H_out
        for x in range(kernel_w):
            x_max = x + stride_w * W_out
            col[:, :, y, x, :, :] = x_padded[:, :, y:y_max:stride_h, x:x_max:stride_w]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
    return col, (H_out, W_out)


def col2im(
    col,
    x_shape,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_top,
    pad_bottom,
    pad_left,
    pad_right,
):
    """
    Inverse of im2col - transforms column matrix back to image.

    Args:
        col: (N * H_out * W_out, C * KH * KW)
        x_shape: original input shape (N, C, H, W)
        kernel_h, kernel_w: kernel dimensions
        stride_h, stride_w: stride values
        pad_*: padding values

    Returns:
        x: (N, C, H, W)
    """
    N, C, H, W = x_shape
    H_padded = H + pad_top + pad_bottom
    W_padded = W + pad_left + pad_right
    H_out = (H_padded - kernel_h) // stride_h + 1
    W_out = (W_padded - kernel_w) // stride_w + 1

    col = col.reshape(N, H_out, W_out, C, kernel_h, kernel_w).transpose(
        0, 3, 4, 5, 1, 2
    )

    x_padded = np.zeros((N, C, H_padded, W_padded))

    for y in range(kernel_h):
        y_max = y + stride_h * H_out
        for x in range(kernel_w):
            x_max = x + stride_w * W_out
            x_padded[:, :, y:y_max:stride_h, x:x_max:stride_w] += col[:, :, y, x, :, :]

    # Remove padding
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        return x_padded[
            :, :, pad_top : H_padded - pad_bottom, pad_left : W_padded - pad_right
        ]
    else:
        return x_padded


class Conv2DFunction(Function):
    """
    VECTORIZED 2D Convolution using im2col for speed.
    This is 100-1000x faster than the loop version!
    """

    def forward(self, x, w, b, stride=(1, 1), padding="valid", padding_mode="constant"):
        # Normalize stride to tuple
        if isinstance(stride, int):
            stride = (stride, stride)
        stride_h, stride_w = stride

        N, C_in, H_in, W_in = x.shape
        C_out, _, KH, KW = w.shape

        # --- Handle padding ---
        if padding == "same":
            pad_top, pad_bottom = calculate_same_padding(H_in, KH, stride_h)
            pad_left, pad_right = calculate_same_padding(W_in, KW, stride_w)
        elif padding == "valid":
            pad_top = pad_bottom = pad_left = pad_right = 0
        elif isinstance(padding, int):
            pad_top = pad_bottom = pad_left = pad_right = padding
        elif isinstance(padding, (tuple, list)):
            if len(padding) == 2:
                pad_top = pad_bottom = padding[0]
                pad_left = pad_right = padding[1]
            elif len(padding) == 4:
                pad_top, pad_bottom, pad_left, pad_right = padding
            else:
                raise ValueError("Padding tuple must be length 2 or 4")
        else:
            raise ValueError(f"Invalid padding: {padding}")

        # Use im2col for fast convolution
        col, (H_out, W_out) = im2col(
            x, KH, KW, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right
        )

        # Reshape weights for matrix multiplication
        w_col = w.reshape(C_out, -1).T  # (C_in * KH * KW, C_out)

        # Fast convolution via matrix multiplication
        out = col @ w_col + b  # (N * H_out * W_out, C_out)

        # Reshape output
        out = out.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)

        # Save for backward (save shapes and needed data)
        self.x_shape = x.shape
        self.w = w  # Need to save w for backward
        self.w_shape = w.shape
        self.col = col
        self.stride = stride
        self.padding_info = (pad_top, pad_bottom, pad_left, pad_right)
        self.out_shape = (H_out, W_out)

        return out

    def backward(self, grad_output):
        stride_h, stride_w = self.stride
        pad_top, pad_bottom, pad_left, pad_right = self.padding_info
        H_out, W_out = self.out_shape

        N, C_out, _, _ = grad_output.shape

        KH, KW = self.w_shape[2], self.w_shape[3]

        # Reshape grad_output
        grad_output_reshaped = grad_output.transpose(0, 2, 3, 1).reshape(-1, C_out)

        # Gradient w.r.t bias (sum over all positions)
        grad_b = np.sum(grad_output_reshaped, axis=0)

        # Gradient w.r.t weights
        grad_w = (self.col.T @ grad_output_reshaped).T.reshape(self.w_shape)

        # Gradient w.r.t input
        w_col = self.w.reshape(C_out, -1).T
        grad_col = grad_output_reshaped @ w_col.T

        grad_x = col2im(
            grad_col,
            self.x_shape,
            KH,
            KW,
            stride_h,
            stride_w,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
        )

        return grad_x, grad_w, grad_b
