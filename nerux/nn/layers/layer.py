from .base import Base
from ...tensor import Tensor

class Layer(Base):
    def __init__(self):
        super().__init__()

    def build(self, in_shape):
        pass
        # self._discover_sub_layers()

    def _discover_sub_layers(self):
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(self, attr_name)
                if isinstance(attr, Base):
                    self.add_layer(attr_name, attr)
            except AttributeError:
                continue

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if isinstance(value, Tensor):
            if value.requires_grad:
                self._parameters[name] = value

        elif isinstance(value, Base):
            self._sub_layers[name] = value
