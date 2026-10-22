from .base import Base


class Sequential(Base):
    def __init__(self, *layers):
        super().__init__()
        if len(layers) == 1 and isinstance(layers[1], (list, tuple)):
            layers = layers[0]

        self._built = True
        for i, layer in enumerate(layers):
            if not isinstance(layer, Base):
                raise ValueError(
                    f"object type {type(layer)} is not an instance of layer class"
                    "provide valid layers objects inside sequential's array "
                )
            self.add_layer(str(i), layer)

    def forward(self, x):
        for layer in self._sub_layers.values():
            x = layer(x)
        return x
