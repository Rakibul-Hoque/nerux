import numpy as np
from .global_grad import Global_grad


class Function:
    def __init__(self, *inputs, **kwargs):
        self.inputs = inputs
        self.kwargs = kwargs
        self.output = None

    @classmethod
    def apply(cls, *inputs, **kwargs):
        from .tensor import Tensor

        global _grad_enabled

        ctx = cls(*inputs, **kwargs)

        ctx.input_versions = []
        inputs_data = []

        for i in inputs:
            if isinstance(i, Tensor):
                inputs_data.append(i.data)
                ctx.input_versions.append(i._version)
            else:
                inputs_data.append(i)
                ctx.input_versions.append(None)

        out_data = ctx.forward(*inputs_data)

        if not isinstance(out_data, np.ndarray):
            out_data = np.array(out_data)

        requires_grad = Global_grad.is_grad_enabled() and any(
            getattr(i, "requires_grad", False) for i in inputs
        )
        creator = ctx if requires_grad else None
        out = Tensor(out_data, requires_grad=requires_grad, creator=creator)

        ctx.output = out
        return out

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError
