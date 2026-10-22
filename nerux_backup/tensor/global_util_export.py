from .global_grad import No_grad, Global_grad
from .utils import export_graph as export_tensor_graph, make_grad as mkgrd
from .checkpoint import Checkpoint


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

def make_grad(func, argnums=None, value=None):
    return mkgrd(func, argnums=argnums, value=value)
__all__ = [name for name in globals() if not name.startswith("_")]
