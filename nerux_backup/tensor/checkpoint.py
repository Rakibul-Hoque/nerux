from .function import Function
from .global_grad import Global_grad
import numpy as np


class Checkpoint(Function):
    @classmethod
    def apply(cls, run_function, *args):
        ctx = cls.__new__(cls)
        ctx.inputs = args
        ctx.kwargs = {}
        ctx.run_function = run_function
        ctx.input_versions = [a._version for a in args]

        Global_grad.set_grad_enabled(False)
        out_tensor = run_function(*args)
        Global_grad.restore_grad_enabled()

        from .tensor import Tensor

        requires_grad = any(getattr(i, "requires_grad", False) for i in args)
        out = Tensor(out_tensor.data, requires_grad=requires_grad, creator=ctx)
        ctx.output = out
        return out

    def forward(self, *args):
        pass

    def backward(self, grad_output):
        from .tensor import Tensor

        detached = [
            Tensor(np.array(inp.data, copy=True), requires_grad=inp.requires_grad)
            for inp in self.inputs
        ]

        Global_grad.set_grad_enabled(True)
        output = self.run_function(*detached)
        Global_grad.restore_grad_enabled()

        output.backward(grad_output)

        return tuple(
            inp.grad.data if inp.grad is not None else np.zeros_like(inp.data)
            for inp in detached
        )
