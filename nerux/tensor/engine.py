import numpy as np
from collections import deque


def build_topo(tensor):
    visited, topo = set(), []

    def visit(t):
        from .tensor import Tensor

        if not isinstance(t, Tensor):
            return
        if t not in visited:
            visited.add(t)
            if t.creator is not None:
                for parent in t.creator.inputs:
                    visit(parent)
            topo.append(t)

    visit(tensor)
    return topo





def run_backward(output_tensor, grad=None):
    from .tensor import Tensor

    if not output_tensor.requires_grad:
        return
    if grad is None:
        grad = np.ones_like(output_tensor.data)
    elif isinstance(grad, Tensor):
        grad = grad.data

    output_tensor.grad = grad
    topo = build_topo(output_tensor)

    for t in reversed(topo):
        if t.creator is None or t.grad is None:
            continue
        creator = t.creator
        for inp, saved_version in zip(creator.inputs, creator.input_versions):
            if isinstance(inp, Tensor) and saved_version is not None:
                if inp._version != saved_version:
                    raise RuntimeError(
                        "In-place operation modified a tensor needed for backward!"
                    )

        grad_out = t.grad.data if isinstance(t.grad, Tensor) else t.grad

        grads = creator.backward(grad_out)

        if not isinstance(grads, tuple):
            grads = (grads,)

        for inp, g in zip(creator.inputs, grads):
            if g is None or not inp.requires_grad:
                continue
            if not isinstance(g, Tensor):
                g = Tensor(g, requires_grad=False)
            if inp.grad is None:
                inp.grad = g
            else:
                assert inp.grad.shape == g.shape, (
                    f"Grad shape mismatch {inp.grad.shape} vs {g.shape}"
                )
                inp.grad.data += g.data
