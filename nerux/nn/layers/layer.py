from .base import Base


class Layer(Base):
    def __init__(self):
        super().__init__()

    def build(self, in_shape):
        self._discover_sub_layers()

    def _discover_sub_layers(self):
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(self, attr_name)
                if isinstance(attr, Base):
                    self.add_layer(attr_name, attr)
            except:
                continue
