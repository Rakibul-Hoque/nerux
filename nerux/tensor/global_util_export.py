from .checkpoint import Checkpoint
from .global_grad import No_grad, Global_grad
from .utils.utils import export_graph as export_tensor_graph
from .utils.grad_fn import make_grad
from .utils.serialization import (
    encode_tensor,
    decode_tensor,
    load_dict,
    save_dict,
    save,
    load,
)


def stop_global_grad():
    Global_grad.stop_global_grad()


def release_global_grad():
    Global_grad.release_global_grad()


def no_grad():
    return No_grad()


def export_graph(tensor, file, as_image=False):
    return export_tensor_graph(tensor, file, as_image)


def checkpoint(fn, args):
    if isinstance(args, tuple):
        return Checkpoint.apply(fn, *args)
    return Checkpoint.apply(fn, args)


__all__ = [name for name in globals() if not name.startswith("_")]
