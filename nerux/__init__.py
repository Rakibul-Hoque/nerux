from .tensor import Tensor, function
from . import nn
from .nn import layers, losses, optim, schedulers, Model
from . import data

nrx = Tensor

__all__ = ["nrx", "Tensor", "function", "nn", "layers", "losses", "schedulers", "optim"]
