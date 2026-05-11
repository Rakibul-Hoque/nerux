import numpy as np
from .tensor import Tensor


def _normalize_shape(shape):
    if len(shape) == 0:
        return None
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])

    return tuple(shape)


# Basic Creation


def tensor(data, dtype=None, requires_grad=False):
    return Tensor(data, dtype=dtype, requires_grad=requires_grad)


def array(data, dtype=None, requires_grad=False):
    return Tensor(data, dtype=dtype, requires_grad=requires_grad)


def from_numpy(arr, requires_grad=False):
    return Tensor(np.array(arr), requires_grad=requires_grad)


# Constant Tensors


def zeros(*shape, dtype=None, requires_grad=False):
    shape = _normalize_shape(shape)
    if shape is None:
        data = np.zeros(())
    else:
        data = np.zeros(shape)
    return Tensor(
        data.astype(dtype or np.float32),
        requires_grad=requires_grad,
    )


def ones(*shape, dtype=None, requires_grad=False):
    shape = _normalize_shape(shape)
    if shape is None:
        data = np.ones(())
    else:
        data = np.ones(shape)
    return Tensor(
        data.astype(dtype or np.float32),
        requires_grad=requires_grad,
    )


def full(*shape, fill_value, dtype=None, requires_grad=False):
    shape = _normalize_shape(shape)
    if shape is None:
        data = np.full((), fill_value)
    else:
        data = np.full(shape, fill_value)
    return Tensor(
        data.astype(dtype or np.float32),
        requires_grad=requires_grad,
    )


def eye(n, dtype=None, requires_grad=False):
    return Tensor(np.eye(n, dtype=dtype or np.float32), requires_grad=requires_grad)


# Like Creation


def zeros_like(x, requires_grad=False):
    return Tensor(np.zeros_like(x.data), requires_grad=requires_grad)


def ones_like(x, requires_grad=False):
    return Tensor(np.ones_like(x.data), requires_grad=requires_grad)


def full_like(x, fill_value, requires_grad=False):
    return Tensor(np.full_like(x.data, fill_value), requires_grad=requires_grad)


# Random


def rand(*shape, dtype=None, requires_grad=False):
    shape = _normalize_shape(shape)
    if shape is None:
        data = np.random.rand()
    else:
        data = np.random.rand(*shape)
    return Tensor(
        np.array(data, dtype=dtype or np.float32),
        requires_grad=requires_grad,
    )


def randn(*shape, dtype=None, requires_grad=False):
    shape = _normalize_shape(shape)

    if shape is None:
        data = np.random.randn()
    else:
        data = np.random.randn(*shape)

    return Tensor(
        np.array(data, dtype=dtype or np.float32),
        requires_grad=requires_grad,
    )


def randint(
    low,
    high=None,
    shape=None,
    dtype=None,
    requires_grad=False,
):
    if high is None:
        low, high = 0, low

    if shape is not None:
        if isinstance(shape, int):
            shape = (shape,)
        elif isinstance(shape, list):
            shape = tuple(shape)
    data = np.random.randint(low, high, size=shape)

    return Tensor(
        np.array(data, dtype=dtype or np.int32),
        requires_grad=requires_grad,
    )


def uniform(
    low=0.0,
    high=1.0,
    *shape,
    dtype=None,
    requires_grad=False,
):
    shape = _normalize_shape(shape)
    data = np.random.uniform(low, high, size=shape)
    return Tensor(
        np.array(data, dtype=dtype or np.float32),
        requires_grad=requires_grad,
    )


# Structured


def arange(start, end=None, step=1, dtype=None, requires_grad=False):
    if end is None:
        start, end = 0, start

    return Tensor(
        np.arange(start, end, step, dtype=dtype or np.float32),
        requires_grad=requires_grad,
    )


def linspace(start, end, steps, dtype=None, requires_grad=False):
    return Tensor(
        np.linspace(start, end, steps, dtype=dtype or np.float32),
        requires_grad=requires_grad,
    )


# Meshgrid


def meshgrid(*tensors, indexing="xy"):
    tensors = [t.data if isinstance(t, Tensor) else t for t in tensors]

    grids = np.meshgrid(*tensors, indexing=indexing)

    return tuple(Tensor(g) for g in grids)


# Triangle


def tril(x, diagonal=0):
    return Tensor(np.tril(x.data, k=diagonal))


def triu(x, diagonal=0):
    return Tensor(np.triu(x.data, k=diagonal))


# Utils


def seed(seed_value):
    np.random.seed(seed_value)


# Exports


__all__ = [name for name in globals() if not name.startswith("_")]
