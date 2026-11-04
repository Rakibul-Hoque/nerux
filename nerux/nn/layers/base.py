from collections import OrderedDict
import json
from ...tensor import Tensor
from ...tensor import functional as F
from ...tensor import factory as init
from ..parameter import Parameter
from ...tensor.utils.json_utils import  _numel

class Base:
    def __init__(self):
        self.training = True
        self._built = False
        self._input_shape = None
        self._version = 1
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._sub_layers = OrderedDict()

    def parameters(self):
        if not self._built:
            raise ValueError(
                "Build the model first (call model.input(...) or model(x))."
            )
        params = list(self._parameters.values())
        for layer in self._sub_layers.values():
            params.extend(layer.parameters())
        return params

    def buffers(self):
        bufs = list(self._buffers.values())
        for layer in self._sub_layers.values():
            bufs.extend(layer.buffers())
        return bufs

    def named_parameters(self, prefix=""):
        for name, p in self._parameters.items():
            yield (f"{prefix}.{name}" if prefix else name), p
        for name, layer in self._sub_layers.items():
            yield from layer.named_parameters(f"{prefix}.{name}" if prefix else name)

    def named_buffers(self, prefix=""):
        for name, b in self._buffers.items():
            yield (f"{prefix}.{name}" if prefix else name), b
        for name, layer in self._sub_layers.items():
            yield from layer.named_buffers(f"{prefix}.{name}" if prefix else name)

    def _iter_state(self, prefix=""):
        for name, p in self._parameters.items():
            key = f"{prefix}.{name}" if prefix else name
            yield key, p
        for name, b in self._buffers.items():
            key = f"{prefix}.{name}" if prefix else name
            yield key, b
        for name, layer in self._sub_layers.items():
            key = f"{prefix}.{name}" if prefix else name
            yield from layer._iter_state(key)

    def state_dict(self, prefix=""):
        return OrderedDict((k, t) for k, t in self._iter_state(prefix))

    def _module_versions(self, prefix=""):
        versions = {prefix or "root": getattr(self, "_version", 1)}
        for name, layer in self._sub_layers.items():
            key = f"{prefix}.{name}" if prefix else name
            versions.update(layer._module_versions(key))
        return versions

    def _module_summary(self, prefix=""):
        name = prefix or "root"

        own_params = list(self._parameters.values())
        own_buffers = list(self._buffers.values())

        entry = {
            "class": type(self).__qualname__,
            "version": getattr(self, "_version", 1),
            "params": len(own_params),
            "param_elements": sum(_numel(p) for p in own_params),
            "buffers": len(own_buffers),
            "buffer_elements": sum(_numel(b) for b in own_buffers),
            "children": list(self._sub_layers.keys()),
        }

        out = {name: entry}

        for child_name, layer in self._sub_layers.items():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            out.update(layer._module_summary(child_prefix))

        return out

    def load_state_dict(self, sd, strict=True, prefix="", _top=True):
        used, expected = set(), set()

        for name, current in self._parameters.items():
            key = f"{prefix}.{name}" if prefix else name
            expected.add(key)
            if key not in sd:
                continue
            new = sd[key]
            if current.shape != new.shape:
                if strict:
                    raise ValueError(
                        f"shape mismatch at {key}: "
                        f"expected {current.shape}, got {new.shape}"
                    )
                continue
            new_param = Parameter.from_tensor(new, requires_grad=new.requires_grad)
            self._parameters[name] = new_param
            setattr(self, name, new_param)
            used.add(key)

        for name, current in self._buffers.items():
            key = f"{prefix}.{name}" if prefix else name
            expected.add(key)
            if key not in sd:
                continue
            new = sd[key]
            if current.shape != new.shape:
                if strict:
                    raise ValueError(
                        f"shape mismatch at {key}: "
                        f"expected {current.shape}, got {new.shape}"
                    )
                continue
            self._buffers[name] = new
            setattr(self, name, new)
            used.add(key)

        for name, layer in self._sub_layers.items():
            key = f"{prefix}.{name}" if prefix else name
            r = layer.load_state_dict(sd, strict=strict, prefix=key, _top=False)
            used |= r["used"]
            expected |= r["expected"]

        if strict and _top:
            missing = expected - used
            unexpected = set(sd) - expected
            if missing:
                raise ValueError(f"missing keys: {sorted(missing)}")
            if unexpected:
                raise ValueError(f"unexpected keys: {sorted(unexpected)}")

        return {"used": used, "expected": expected}

    def add_parameter(self, name, value):
        if not isinstance(value, Parameter):
            raise ValueError("Parameter must be an instance of Parameter.")
        self._parameters[name] = value
        setattr(self, name, value)
        return value

    def add_buffer(self, name, value):
        if not isinstance(value, Tensor):
            value = init.tensor(value)
        self._buffers[name] = value
        setattr(self, name, value)
        return value

    def add_layer(self, name, layer):
        if not isinstance(layer, Base):
            raise ValueError("Sub layer must be an instance of Base.")
        self._sub_layers[name] = layer
        setattr(self, name, layer)
        return layer

    def set_training(self, value):
        self.training = value
        for layer in self._sub_layers.values():
            layer.set_training(value)

    def input(self, *in_shape):
        self._input_shape = in_shape
        dummy_input = init.zeros((1,) + tuple(in_shape))
        self(dummy_input)
        return self

    def __call__(self, x, *args, **kwargs):
        if not self._built:
            if hasattr(self, "build"):
                input_shape = x.shape[1:]
                self.build(input_shape)
            self._built = True

        return self.forward(x, *args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward() must be implemented in subclass")
