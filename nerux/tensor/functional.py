# nerux/functional.py

from .tensor import Tensor

__all__ = []

# Methods that should NOT become module-level functions
SKIP = {
    # python internals
    "__init__",
    "__repr__",
    "__getitem__",
    "__setitem__",
    "__len__",
    "__array__",
    "__hash__",
    # properties / utility
    "shape",
    "dtype",
    "ndim",
    "grad_fn",
    "is_leaf",
    "T",
    # graph / state management
    "backward",
    "zero_grad",
    "clip_grad",
    "requires_grad_",
    "_bump_version",
    "_check_inplace_safe",
    # inplace ops
    "add_",
    "sub_",
    "mul_",
    "div_",
    "zero_",
    # conversions
    "item",
    "tolist",
    "to_numpy",
    "detach",
    "clone",
    "copy",
    # visualization
    "print",
    "show_graph",
    "get_graph",
    # static creation methods
    "tensor",
    "array",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "randn",
    "rand",
    "randint",
    "uniform",
    "arange",
    "linspace",
    "eye",
    "full",
    "full_like",
    # static combine methods
    "concat",
    "stack",
    "meshgrid",
    # global grad
    "no_grad",
    "stop_global_grad",
    "release_global_grad",
    # exports
    "export_graph",
    "seed",
    # special static methods
    "checkpoint",
}


def _make_wrapper(name):
    def wrapper(x, *args, **kwargs):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        method = getattr(x, name)
        return method(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__doc__ = f"Functional wrapper for Tensor.{name}()"

    return wrapper


# Automatically expose Tensor methods as module-level functions
for name in dir(Tensor):
    if name.startswith("_"):
        continue

    if name in SKIP:
        continue

    attr = getattr(Tensor, name)

    if callable(attr):
        globals()[name] = _make_wrapper(name)
        __all__.append(name)
