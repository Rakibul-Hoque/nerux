import numpy as np


def _clip_grad_norm(params, max_norm, norm_type=2.0):
    """
    Clip gradients by global L2 norm.
    All parameter gradients are treated as one long vector —
    the norm of that vector is computed, then every gradient is
    scaled down proportionally if the norm exceeds max_norm.

    Returns the global norm BEFORE clipping (useful for monitoring).
    """
    grads = [p.grad.data for p in params if p.grad is not None]
    if not grads:
        return 0.0

    if norm_type == float("inf"):
        global_norm = max(np.abs(g).max() for g in grads)
    else:
        global_norm = np.sqrt(sum(np.sum(g**norm_type) for g in grads))

    if global_norm > max_norm:
        scale = max_norm / (global_norm + 1e-8)
        for p in params:
            if p.grad is not None:
                p.grad.data *= scale

    return float(global_norm)


def _clip_grad_value(params, clip_value):
    """
    Clip each gradient element-wise to [-clip_value, clip_value].
    Less common than norm clipping — used when you care about
    individual parameter scales rather than the global update direction.
    """
    for p in params:
        if p.grad is not None:
            np.clip(p.grad.data, -clip_value, clip_value, out=p.grad.data)


def clip_grad_norm(params, max_norm, norm_type=2.0):
    return _clip_grad_norm(list(params), max_norm, norm_type)


def clip_grad_value(params, clip_value):
    _clip_grad_value(list(params), clip_value)
