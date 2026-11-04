from .base import Base
from ...tensor import Tensor
from ..parameter import Parameter


class Layer(Base):
    def __init__(self):
        super().__init__()

    def build(self, in_shape):
        pass

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if isinstance(value, Tensor):
            self._buffers[name] = value
        elif isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Base):
            self._sub_layers[name] = value
