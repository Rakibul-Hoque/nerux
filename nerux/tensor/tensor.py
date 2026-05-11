import numpy as np
import time

from .engine import run_backward
from .global_grad import No_grad, Global_grad

from .utils import (
    Tid_count,
    print_tensor,
    get_graph,
    _format_time,
    export_graph,
)


from . import functional as F


def _ensure_numpy(data, dt=None):
    if isinstance(data, Tensor):
        return data.data.astype(dt) if dt else data.data

    if isinstance(data, memoryview):
        data = np.array(data)

    if not isinstance(data, np.ndarray):
        return np.array(data, dtype=dt or np.float32)

    return data.astype(dt) if dt else data


class Tensor:
    __slots__ = (
        "data",
        "requires_grad",
        "creator",
        "grad",
        "_version",
        "_base",
        "_is_view",
        "_id",
    )

    # Init

    def __init__(
        self,
        data,
        dtype=None,
        requires_grad=False,
        creator=None
    ):
        self.data = _ensure_numpy(data, dtype)

        self.requires_grad = requires_grad
        self.creator = creator

        self.grad = None

        self._version = 0

        self._base = None
        self._is_view = False

        self._id = Tid_count.get_incremented()

    # Autograd

    def backward(self, grad=None, verbose=False):
        start = time.perf_counter()
        run_backward(self, grad)
        end = time.perf_counter()

        if verbose:
            print(f"Backward pass took {_format_time(end - start)}")

    def zero_grad(self):
        self.grad = None

    def clip_grad(self, max_norm):
        if self.grad is None:
            return

        norm = np.linalg.norm(self.grad)

        if norm > max_norm:
            self.grad.data *= max_norm / (norm + 1e-8)

    def requires_grad_(self, flag=True):
        self.requires_grad = flag
        return self

    # Properties

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def is_leaf(self):
        return self.creator is None

    @property
    def grad_fn(self):
        return self.creator

    @property
    def T(self):
        return F.transpose(self)

    # Conversion

    def item(self):
        assert self.data.size == 1
        return self.data.item()

    def to_numpy(self):
        return self.data.copy()

    def tolist(self):
        return self.data.copy().tolist()

    def clone(self):
        return Tensor(
            self.data.copy(),
            requires_grad=self.requires_grad,
        )

    def copy(self):
        return self.clone()

    def detach(self):
        return Tensor(
            self.data.copy(),
            requires_grad=False,
        )

    def astype(self, dtype):
        return Tensor(
            self.data.astype(dtype),
            requires_grad=self.requires_grad,
        )

    def float(self):
        return self.astype(np.float32)

    def int(self):
        return self.astype(np.int32)

    # Array Protocol

    def __array__(self, dtype=None):
        if dtype is None:
            return self.data

        return self.data.astype(dtype)

    def __hash__(self):
        return id(self)

    def __len__(self):
        return len(self.data)

    # Arithmetic Operators

    def __add__(self, other):
        return F.add(self, other)

    def __mul__(self, other):
        return F.mul(self, other)

    def __sub__(self, other):
        return F.sub(self, other)

    def __truediv__(self, other):
        return F.div(self, other)

    def __neg__(self):
        return F.neg(self)

    def __pow__(self, power):
        return F.pow(self, power)

    def __matmul__(self, other):
        return F.matmul(self, other)

    def __invert__(self):
        return F.logical_not(self)

    # Reverse Operators

    def __radd__(self, other):
        return F.add(other, self)

    def __rsub__(self, other):
        return F.sub(other, self)

    def __rmul__(self, other):
        return F.mul(other, self)

    def __rtruediv__(self, other):
        return F.div(other, self)

    def __rpow__(self, other):
        return F.pow(other, self)

    # Comparison

    def __eq__(self, other):
        other = F.ensure_tensor(other)
        return Tensor(self.data == other.data)

    def __gt__(self, other):
        other = F.ensure_tensor(other)
        return Tensor((self.data > other.data).astype(np.float32))

    def __lt__(self, other):
        other = F.ensure_tensor(other)
        return Tensor((self.data < other.data).astype(np.float32))

    def __ge__(self, other):
        other = F.ensure_tensor(other)
        return Tensor((self.data >= other.data).astype(np.float32))

    def __le__(self, other):
        other = F.ensure_tensor(other)
        return Tensor((self.data <= other.data).astype(np.float32))

    # Indexing

    def __getitem__(self, idx):
        out = F.getitem(self, idx)

        out._base = self
        out._is_view = True
        out._version = self._version

        return out

    def __setitem__(self, idx, value):
        value = F.ensure_tensor(value)

        self._check_inplace_safe()

        self.data[idx] = value.data

        self._bump_version()

    # Arithmetic Methods

    def sum(self, axis=None, keepdims=False):
        return F.sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return F.mean(self, axis, keepdims)

    def std(self, axis=None, keepdims=False):
        return F.std(self, axis, keepdims)

    def min(self, other=None, axis=None):
        return F.min(self, other, axis)

    def max(self, other=None, axis=None):
        return F.max(self, other, axis)

    def pow(self, power):
        return F.pow(self, power)

    def sqrt(self):
        return F.sqrt(self)

    def exp(self):
        return F.exp(self)

    def log(self):
        return F.log(self)

    # Activations

    def relu(self):
        return F.relu(self)

    def sigmoid(self):
        return F.sigmoid(self)

    def tanh(self):
        return F.tanh(self)

    def softplus(self):
        return F.softplus(self)

    def elu(self, alpha=0.01):
        return F.elu(self, alpha)

    def leakyrelu(self, alpha=0.01):
        return F.leakyrelu(self, alpha)

    def softmax(self, axis=-1):
        return F.softmax(self, axis)

    def log_softmax(self, axis=-1):
        return F.log_softmax(self, axis)

    # Shape

    def reshape(self, *shape):
        return F.reshape(self, *shape)

    def transpose(self, axes=None):
        return F.transpose(self, axes)

    def permute(self, *axes):
        return F.permute(self, *axes)

    def squeeze(self, axis=None):
        return F.squeeze(self, axis)

    def unsqueeze(self, axis):
        return F.unsqueeze(self, axis)

    def flatten(self):
        return F.flatten(self)

    def clip(self, min_value, max_value):
        return F.clip(self, min_value, max_value)

    # Indexing Ops

    def gather(self, axis, index):
        return F.gather(self, axis, index)

    def scatter(self, axis, index, shape):
        return F.scatter(self, axis, index, shape)

    def masked_fill(self, mask, fill_value):
        return F.masked_fill(self, mask, fill_value)

    # Tensor Manipulation

    def tile(self, reps):
        return F.tile(self, reps)

    def repeat(self, repeats, axis=None):
        return F.repeat(self, repeats, axis)

    def roll(self, shift, axis=None):
        return F.roll(self, shift, axis)

    def flip(self, axis=None):
        return F.flip(self, axis)

    def pad(self, pad_width, mode="constant", constant_value=0):
        return F.pad(self, pad_width, mode, constant_value)

    # Statistics

    def cumsum(self, axis=None):
        return F.cumsum(self, axis)

    def cumprod(self, axis=None):
        return F.cumprod(self, axis)

    def argmax(self, axis=None):
        return F.argmax(self, axis)

    def argmin(self, axis=None):
        return F.argmin(self, axis)

    def topk(self, k, axis=-1, largest=True):
        return F.topk(self, k, axis, largest)

    # Linear Algebra

    def dot(self, other):
        return F.dot(self, other)

    def outer(self, other):
        return F.outer(self, other)

    def norm(self, ord=2, axis=None, keepdims=False):
        return F.norm(self, ord, axis, keepdims)

    def det(self):
        return F.det(self)

    def inv(self):
        return F.inv(self)

    def trace(self):
        return F.trace(self)

    def diag(self):
        return F.diag(self)

    def svd(self):
        return F.svd(self)

    # Scientific

    def abs(self):
        return F.abs(self)

    def sign(self):
        return F.sign(self)

    def sin(self):
        return F.sin(self)

    def cos(self):
        return F.cos(self)

    def tan(self):
        return F.tan(self)

    def arcsin(self):
        return F.arcsin(self)

    def arccos(self):
        return F.arccos(self)

    def arctan(self):
        return F.arctan(self)

    def sinh(self):
        return F.sinh(self)

    def cosh(self):
        return F.cosh(self)

    def log2(self):
        return F.log2(self)

    def log10(self):
        return F.log10(self)

    def log1p(self):
        return F.log1p(self)

    def expm1(self):
        return F.expm1(self)

    def ceil(self):
        return F.ceil(self)

    def floor(self):
        return F.floor(self)

    def round(self, decimals=0):
        return F.round(self, decimals)

    def erf(self):
        return F.erf(self)

    # Views

    def unbind(self, axis=0):
        n = self.shape[axis]

        return tuple(
            self[tuple(i if j == axis else slice(None) for j in range(self.ndim))]
            for i in range(n)
        )

    # Inplace Safety

    def _bump_version(self):
        self._version += 1

        if self._base is not None:
            self._base._version += 1

    def _check_inplace_safe(self):
        if self.requires_grad and not self.is_leaf and Global_grad.is_grad_enabled():
            raise RuntimeError(
                "In-place operation on non-leaf tensor that requires grad"
            )

        if self._is_view:
            raise RuntimeError("In-place operation on a view is not allowed")

    # Inplace Ops

    @property
    def value(self):
        return self.data.copy()

    @value.setter
    def value(self, value):
        self._check_inplace_safe()

        if isinstance(value, Tensor):
            value = value.data

        elif not isinstance(value, np.ndarray):
            value = np.array(value)

        self.data = value

        self._bump_version()

    def zero_(self):
        self.data.fill(0)
        self._bump_version()
        return self

    def add_(self, other):
        other = F.ensure_tensor(other)
        self.value = self.data + other.data
        return self

    def sub_(self, other):
        other = F.ensure_tensor(other)
        self.value = self.data - other.data
        return self

    def mul_(self, other):
        other = F.ensure_tensor(other)
        self.value = self.data * other.data
        return self

    def div_(self, other):
        other = F.ensure_tensor(other)
        self.value = self.data / other.data
        return self

    # Visualization

    def get_graph(self, max_depth=None, use_color=True):
        return get_graph(self, max_depth=max_depth, use_color=use_color)

    def show_graph(self, max_depth=None, use_color=True):
        print(self.get_graph(max_depth=max_depth, use_color=use_color))

    def print(self):
        print(self.tolist())

    def __repr__(self):
        return print_tensor(self)
