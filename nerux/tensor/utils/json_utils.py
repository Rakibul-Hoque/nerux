from .. import factory as init
from .. import functional as F
import functools, operator

def _tensor_to_json(t):
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "requires_grad": t.requires_grad,
        "data": t.tolist(),
    }


def _json_to_tensor(blob):
    t = init.tensor(
        blob["data"], dtype=F.dtype(blob["dtype"]), requires_grad=blob["requires_grad"]
    )

    if tuple(blob["shape"]) != tuple(t.shape):
        raise ValueError(
            f"serialized shape {blob['shape']} != reconstructed {tuple(t.shape)}"
        )
    return t


def _numel(t):
    shape = tuple(t.shape)
    return functools.reduce(operator.mul, shape, 1) if shape else 1
