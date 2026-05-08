from collections import OrderedDict
from ...tensor import Tensor


class Base:
    def __init__(self):
        self.training = True
        self._built = False
        self._parameters = OrderedDict()
        self._sub_layers = OrderedDict()

    def parameters(self):
        if not self._built:
            raise ValueError(
                "You must set an input shape e.g. 'model.input(10,50)' or call the model  e.g. 'model(x)' to initialize the parameters "
            )
        params = list(self._parameters.values())
        for layer in self._sub_layers.values():
            params.extend(layer.parameters())
        return params

    def parameters_dict(self):
        own = {
            name: value.tolist() if isinstance(value, Tensor) else value
            for name, value in self._parameters.items()
        }

        subs = {
            name: layer.parameters_dict() for name, layer in self._sub_layers.items()
        }
        dic = {}
        if self._parameters:
            dic["self"] = own
        if self._sub_layers:
            dic["subs"] = subs
        return dic

    def set_parameters(self, dic):
        own = dic.get("self", None)
        if own:
            if self._parameters.keys() != own.keys():
                raise ValueError("Layer parameters Doesn't match given parameters key")
            for key, value in own.items():
                tensor = Tensor.array(value, requires_grad=True)
                self._parameters[key] = tensor
                setattr(self, key, tensor)
        subs = dic.get("subs", None)
        if subs:
            if self._sub_layers.keys() != subs.keys():
                raise ValueError(
                    "sub layer Doesn't match given sub layers in parameters"
                )
            for name, layer in self._sub_layers.items():
                layer.set_parameters(subs[name])

    def set_training(self, value):
        self.training = value
        for layer in self._sub_layers.values():
            layer.set_training(value)

    def add_parameter(self, name, value):
        if not isinstance(value, Tensor):
            raise ValueError("Parameter must be tensor object")
        value.requires_grad_(True)
        self._parameters[name] = value
        return value

    def add_layer(self, name, layer):
        if not isinstance(layer, type(self)):
            raise ValueError("Sub layer must be an instance if Baae layer class")

        self._sub_layers[name] = layer
        return layer

    def __call__(self, x, *args, **kwargs):
        if not self._built:
            if hasattr(self, "build"):
                input_shape = x.shape[1:]
                self.build(input_shape)
            self._built = True

        return self.forward(x, *args, **kwargs)

    def input(self, *in_shape):
        
        dummy_input = Tensor.zeros((1,) + tuple(in_shape))
        self(dummy_input)
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward() must be implemented in subclass")
