# functional/_nd.py
# Pure-NumPy N-d (1D / 2D / 3D) im2col machinery shared by Conv and Pool.
# No autograd dependency: the per-dimension Function classes call into this.
#
# Layout: x is (N, C, *spatial); conv weights are (C_out, C_in, *kernel).

import itertools
import numpy as np

_LEAD = (slice(None), slice(None))


# ----------------------------------------------------------------------------
# argument handling
# ----------------------------------------------------------------------------


def normalize_tuple(value, n, name="value"):
    if isinstance(value, (int, np.integer)):
        return (int(value),) * n
    value = tuple(value)
    if len(value) != n:
        raise ValueError(f"{name} must be an int or a tuple of length {n}, got {value}")
    return tuple(int(v) for v in value)


def calculate_same_padding(input_size, kernel_size, stride):
    """Calculate padding needed for 'same' output size"""
    output_size = (input_size + stride - 1) // stride
    padding_needed = max(0, (output_size - 1) * stride + kernel_size - input_size)
    pad_before = padding_needed // 2
    pad_after = padding_needed - pad_before
    return pad_before, pad_after


def resolve_padding(padding, spatial_shape, kernel, stride):
    """
    -> tuple of (before, after) per spatial dim.
    "same" | "valid" | int | tuple(n) symmetric per dim |
    tuple(2n) as (d0_before, d0_after, d1_before, d1_after, ...)
    """
    n = len(kernel)
    if isinstance(padding, str):
        if padding == "same":
            pads = tuple(
                calculate_same_padding(i, k, s)
                for i, k, s in zip(spatial_shape, kernel, stride)
            )
        elif padding == "valid":
            pads = ((0, 0),) * n
        else:
            raise ValueError(f"Invalid padding: {padding}")
    elif isinstance(padding, (int, np.integer)):
        pads = ((int(padding), int(padding)),) * n
    elif isinstance(padding, (tuple, list)):
        if len(padding) == n:
            pads = tuple((int(p), int(p)) for p in padding)
        elif len(padding) == 2 * n:
            pads = tuple((int(padding[2 * i]), int(padding[2 * i + 1])) for i in range(n))
        else:
            raise ValueError(f"Padding tuple must be length {n} or {2 * n}")
    else:
        raise ValueError(f"Invalid padding: {padding}")

    if any(b < 0 or a < 0 for b, a in pads):
        raise ValueError(f"Padding must be non-negative, got {pads}")
    return pads


def check_padding_mode(padding_mode):
    if padding_mode != "constant":
        raise ValueError(
            f"padding_mode={padding_mode!r} is not supported, only 'constant'"
        )


def _float_dtype(x):
    return x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64


def pad_input(x, pads, value=0.0):
    if not any(b > 0 or a > 0 for b, a in pads):
        return x
    return np.pad(
        x, ((0, 0), (0, 0)) + tuple(pads), mode="constant", constant_values=value
    )


def output_shape(padded_shape, kernel, stride, who="op"):
    if any(k <= 0 for k in kernel) or any(s <= 0 for s in stride):
        raise ValueError(f"{who}: kernel and stride must be positive")
    out = tuple((p - k) // s + 1 for p, k, s in zip(padded_shape, kernel, stride))
    if any(o <= 0 for o in out):
        raise ValueError(
            f"{who}: kernel {tuple(kernel)} is larger than padded input "
            f"{tuple(padded_shape)}; output size would be {out}"
        )
    return out


# ----------------------------------------------------------------------------
# windows <-> arrays
# ----------------------------------------------------------------------------


def _slices(idx, stride, out_shape):
    return tuple(slice(i, i + s * o, s) for i, s, o in zip(idx, stride, out_shape))


def extract_windows(x_padded, kernel, stride, out_shape):
    """(N, C, *padded) -> (N, C, *kernel, *out)"""
    N, C = x_padded.shape[:2]
    windows = np.zeros((N, C) + tuple(kernel) + tuple(out_shape), dtype=_float_dtype(x_padded))
    for idx in itertools.product(*(range(k) for k in kernel)):
        windows[_LEAD + idx] = x_padded[_LEAD + _slices(idx, stride, out_shape)]
    return windows


def scatter_windows(windows, padded_shape, kernel, stride):
    """Adjoint of extract_windows (sums overlaps): (N, C, *kernel, *out) -> (N, C, *padded)"""
    n = len(kernel)
    N, C = windows.shape[:2]
    out_shape = windows.shape[2 + n :]
    x_padded = np.zeros((N, C) + tuple(padded_shape), dtype=_float_dtype(windows))
    for idx in itertools.product(*(range(k) for k in kernel)):
        x_padded[_LEAD + _slices(idx, stride, out_shape)] += windows[_LEAD + idx]
    return x_padded


def unpad(x_padded, pads, spatial_shape):
    sl = tuple(slice(b, b + size) for (b, _), size in zip(pads, spatial_shape))
    return x_padded[_LEAD + sl]


def im2col_nd(x, kernel, stride, pads, who="conv"):
    """x (N, C, *spatial) -> col (N * prod(out), C * prod(kernel)), out_shape"""
    n = len(kernel)
    N = x.shape[0]
    x_padded = pad_input(x, pads)
    out_shape = output_shape(x_padded.shape[2:], kernel, stride, who)
    windows = extract_windows(x_padded, kernel, stride, out_shape)
    axes = (0,) + tuple(range(2 + n, 2 + 2 * n)) + (1,) + tuple(range(2, 2 + n))
    col = windows.transpose(axes).reshape(N * int(np.prod(out_shape)), -1)
    return col, out_shape


def col2im_nd(col, x_shape, kernel, stride, pads):
    """Adjoint of im2col_nd -> (N, C, *spatial)"""
    n = len(kernel)
    N, C = x_shape[:2]
    spatial = tuple(x_shape[2:])
    padded = tuple(s + b + a for s, (b, a) in zip(spatial, pads))
    out_shape = output_shape(padded, kernel, stride)

    col = col.reshape((N,) + out_shape + (C,) + tuple(kernel))
    axes = (0, 1 + n) + tuple(range(2 + n, 2 + 2 * n)) + tuple(range(1, 1 + n))
    x_padded = scatter_windows(col.transpose(axes), padded, kernel, stride)
    return unpad(x_padded, pads, spatial)


# ----------------------------------------------------------------------------
# Convolution
# ----------------------------------------------------------------------------


def conv_forward_nd(x, w, b, stride, padding, padding_mode, ndim):
    check_padding_mode(padding_mode)
    x, w, b = np.asarray(x), np.asarray(w), np.asarray(b)
    who = f"Conv{ndim}D"
    if x.ndim != ndim + 2:
        raise ValueError(f"{who} expects {ndim + 2}D input (N, C, ...), got shape {x.shape}")
    if w.ndim != ndim + 2:
        raise ValueError(f"{who} weight must be {ndim + 2}D, got shape {w.shape}")
    if w.shape[1] != x.shape[1]:
        raise ValueError(
            f"{who}: input has {x.shape[1]} channels but weight expects {w.shape[1]}"
        )

    stride = normalize_tuple(stride, ndim, "stride")
    N, C_out = x.shape[0], w.shape[0]
    kernel = tuple(w.shape[2:])
    pads = resolve_padding(padding, x.shape[2:], kernel, stride)

    col, out_shape = im2col_nd(x, kernel, stride, pads, who)
    w_col = w.reshape(C_out, -1).T  # (C_in * prod(K), C_out)
    out = col @ w_col + b  # (N * prod(out), C_out)

    out = out.reshape((N,) + out_shape + (C_out,))
    out = out.transpose((0, ndim + 1) + tuple(range(1, ndim + 1)))

    cache = dict(
        ndim=ndim, x_shape=x.shape, w=w, col=col, stride=stride, pads=pads
    )
    return out, cache


def conv_backward_nd(grad_output, c):
    n = c["ndim"]
    w = c["w"]
    C_out = w.shape[0]
    kernel = tuple(w.shape[2:])

    g = grad_output.transpose((0,) + tuple(range(2, n + 2)) + (1,)).reshape(-1, C_out)
    grad_b = np.sum(g, axis=0)
    grad_w = (c["col"].T @ g).T.reshape(w.shape)
    grad_col = g @ w.reshape(C_out, -1)
    grad_x = col2im_nd(grad_col, c["x_shape"], kernel, c["stride"], c["pads"])
    return grad_x, grad_w, grad_b


# ----------------------------------------------------------------------------
# Pooling
# ----------------------------------------------------------------------------


def pool_forward_nd(x, kernel_size, stride, padding, padding_mode, mode, ndim):
    check_padding_mode(padding_mode)
    if mode not in ("max", "avg"):
        raise ValueError(f"mode must be 'max' or 'avg', got '{mode}'")
    x = np.asarray(x)
    who = f"Pool{ndim}D"
    if x.ndim != ndim + 2:
        raise ValueError(f"{who} expects {ndim + 2}D input (N, C, ...), got shape {x.shape}")

    kernel = normalize_tuple(kernel_size, ndim, "kernel_size")
    stride = kernel if stride is None else normalize_tuple(stride, ndim, "stride")
    N, C = x.shape[:2]
    pads = resolve_padding(padding, x.shape[2:], kernel, stride)

    if mode == "max" and any(b >= k or a >= k for (b, a), k in zip(pads, kernel)):
        raise ValueError(
            f"{who}: max-pool padding {pads} must be smaller than kernel {kernel} "
            "(otherwise a window can contain only padding)"
        )

    x_padded = pad_input(x, pads, -np.inf if mode == "max" else 0.0)
    out_shape = output_shape(x_padded.shape[2:], kernel, stride, who)

    windows = extract_windows(x_padded, kernel, stride, out_shape)
    axes = (0, 1) + tuple(range(2 + ndim, 2 + 2 * ndim)) + tuple(range(2, 2 + ndim))
    col = windows.transpose(axes).reshape((N, C) + out_shape + (-1,))

    cache = dict(
        ndim=ndim, x_shape=x.shape, kernel=kernel, stride=stride, pads=pads, mode=mode
    )
    if mode == "max":
        out = np.max(col, axis=-1)
        cache["max_indices"] = np.argmax(col, axis=-1)
    else:
        out = np.mean(col, axis=-1)
    return out, cache


def pool_backward_nd(grad_output, c):
    n = c["ndim"]
    kernel, stride, pads = c["kernel"], c["stride"], c["pads"]
    N, C = c["x_shape"][:2]
    spatial = tuple(c["x_shape"][2:])
    padded = tuple(s + b + a for s, (b, a) in zip(spatial, pads))
    out_shape = tuple(grad_output.shape[2:])
    K = int(np.prod(kernel))

    if c["mode"] == "max":
        g = np.zeros(grad_output.shape + (K,), dtype=_float_dtype(grad_output))
        np.put_along_axis(g, c["max_indices"][..., None], grad_output[..., None], axis=-1)
        g = g.reshape((N, C) + out_shape + kernel)
        axes = (0, 1) + tuple(range(2 + n, 2 + 2 * n)) + tuple(range(2, 2 + n))
        grad_padded = scatter_windows(g.transpose(axes), padded, kernel, stride)
    else:
        grad_padded = np.zeros((N, C) + padded, dtype=_float_dtype(grad_output))
        gp = grad_output / K
        for idx in itertools.product(*(range(k) for k in kernel)):
            grad_padded[_LEAD + _slices(idx, stride, out_shape)] += gp

    return unpad(grad_padded, pads, spatial)
