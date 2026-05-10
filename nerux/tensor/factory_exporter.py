# nerux/__init__.py

from .tensor import Tensor
from .factory import TensorFactory


tensor = TensorFactory.tensor
array = TensorFactory.array

zeros = TensorFactory.zeros
ones = TensorFactory.ones

zeros_like = TensorFactory.zeros_like
ones_like = TensorFactory.ones_like

randn = TensorFactory.randn
rand = TensorFactory.rand
randint = TensorFactory.randint
uniform = TensorFactory.uniform

arange = TensorFactory.arange
linspace = TensorFactory.linspace
eye = TensorFactory.eye

full = TensorFactory.full
full_like = TensorFactory.full_like


concat = TensorFactory.concat
stack = TensorFactory.stack
meshgrid = TensorFactory.meshgrid

# =========================================================
# utilities
# =========================================================

seed = TensorFactory.seed

no_grad = TensorFactory.no_grad

checkpoint = TensorFactory.checkpoint

export_graph = TensorFactory.export_graph

stop_global_grad = TensorFactory.stop_global_grad
release_global_grad = TensorFactory.release_global_grad

# =========================================================
# exports
# =========================================================

__all__ = [
    # main tensor
    "TensorFactory",
    # creation
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
    # combine
    "concat",
    "stack",
    "meshgrid",
    # utility
    "seed",
    "no_grad",
    "checkpoint",
    "export_graph",
    "stop_global_grad",
    "release_global_grad",

]

