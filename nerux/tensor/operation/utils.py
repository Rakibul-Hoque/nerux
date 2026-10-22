import numpy as np



def ensure_array(data, dtype=None):
    if isinstance(data, np.ndarray):
        return data.astype(dtype) if dtype else data

    return np.array(data, dtype=dtype if dtype else np.float32)


def reduce_grad(grad, shape):
    # Strip extra leading dims
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Sum over broadcasted dims (shape had size 1, grad may have size > 1)
    axes = tuple(i for i, dim in enumerate(shape) if dim == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)

    if shape == ():
        grad = grad.sum()

    assert grad.shape == shape, f"reduce_grad shape mismatch: {grad.shape} vs {shape}"
    return grad
