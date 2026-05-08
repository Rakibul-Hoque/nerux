import numpy as np
from .operation.main_operations import Concat, Stack, Min, Max, Where


class TensorFactory:
    Tensor = None 

    # -------- Basic --------
    @classmethod
    def array(cls, arr, dtype=None, requires_grad=False):
        return cls.Tensor(arr, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def tensor(cls, arr, dtype=None, requires_grad=False):
        return cls.Tensor(arr, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def from_array(cls, arr, requires_grad=False):
        data = arr.data if isinstance(arr, cls.Tensor) else arr
        return cls.Tensor(np.array(data), requires_grad=requires_grad)

    # -------- Zeros / Ones --------
    @classmethod
    def zeros(cls, shape, dtype=None, requires_grad=False):
        return cls.Tensor(
            np.zeros(shape, dtype=dtype or np.float32), requires_grad=requires_grad
        )

    @classmethod
    def ones(cls, shape, dtype=None, requires_grad=False):
        return cls.Tensor(
            np.ones(shape, dtype=dtype or np.float32), requires_grad=requires_grad
        )

    @classmethod
    def zeros_like(cls, x, requires_grad=False):
        data = x.data if isinstance(x, cls.Tensor) else x
        return cls.Tensor(np.zeros_like(data), requires_grad=requires_grad)

    @classmethod
    def ones_like(cls, x, requires_grad=False):
        data = x.data if isinstance(x, cls.Tensor) else x
        return cls.Tensor(np.ones_like(data), requires_grad=requires_grad)

    # -------- Random --------
    @classmethod
    def randn(cls, shape, dtype=None, requires_grad=False):
        return cls.Tensor(
            np.random.randn(*shape).astype(dtype or np.float32),
            requires_grad=requires_grad,
        )

    @classmethod
    def rand(cls, shape, dtype=None, requires_grad=False):
        return cls.Tensor(
            np.random.rand(*shape).astype(dtype or np.float32),
            requires_grad=requires_grad,
        )

    @classmethod
    def randint(cls, low, high, shape, dtype=None, requires_grad=False):
        return cls.Tensor(
            np.random.randint(low, high, shape).astype(dtype or np.int32),
            requires_grad=requires_grad,
        )

    @classmethod
    def uniform(cls, low, high, shape, dtype=None, requires_grad=False):
        return cls.Tensor(
            np.random.uniform(low, high, shape).astype(dtype or np.float32),
            requires_grad=requires_grad,
        )

    # -------- Structured --------
    @classmethod
    def arange(cls, start, end=None, step=1, dtype=None, requires_grad=False):
        if end is None:
            start, end = 0, start
        return cls.Tensor(
            np.arange(start, end, step, dtype=dtype or np.float32),
            requires_grad=requires_grad,
        )

    @classmethod
    def linspace(cls, start, end, steps, dtype=None, requires_grad=False):
        return cls.Tensor(
            np.linspace(start, end, steps, dtype=dtype or np.float32),
            requires_grad=requires_grad,
        )

    @classmethod
    def eye(cls, n, dtype=None, requires_grad=False):
        return cls.Tensor(
            np.eye(n, dtype=dtype or np.float32), requires_grad=requires_grad
        )

    @classmethod
    def full(cls, shape, fill_value, dtype=None, requires_grad=False):
        return cls.Tensor(
            np.full(shape, fill_value, dtype=dtype or np.float32),
            requires_grad=requires_grad,
        )

    @classmethod
    def full_like(cls, x, fill_value, requires_grad=False):
        data = x.data if isinstance(x, cls.Tensor) else x
        return cls.Tensor(np.full_like(data, fill_value), requires_grad=requires_grad)

    # -------- Combine --------
    @classmethod
    def concat(cls, tensors, axis=0):
        return Concat.apply(*tensors, axis=axis)

    @classmethod
    def stack(cls, tensors, axis=0):
        return Stack.apply(*tensors, axis=axis)

    # -------- Elementwise --------
    @classmethod
    def where(cls, cond, x, y):
        return Where.apply(x, y, cond=cond)

    @classmethod
    def minimum(cls, x, y):
        return Min.apply(cls._ensure_tensor(x), cls._ensure_tensor(y))

    @classmethod
    def maximum(cls, x, y):
        return Max.apply(cls._ensure_tensor(x), cls._ensure_tensor(y))

    # -------- Utils --------
    @classmethod
    def seed(cls, seed):
        np.random.seed(seed)

    @classmethod
    def _ensure_tensor(cls, x):
        if isinstance(x, cls.Tensor):
            return x
        return cls.Tensor(x)


