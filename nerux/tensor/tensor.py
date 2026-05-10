# tensor.py
import numpy as np
from .engine import run_backward
from .factory import TensorFactory

from .global_grad import No_grad, Global_grad

from .utils import (
    Tid_count,
    print_tensor,
    get_graph,
    _format_time,
    TensorView,
    export_graph,
)
from .checkpoint import Checkpoint
import time
from .operation.main_operations import (
    Add,
    Mul,
    Concat,
    MatMul,
    Transpose,
    Reshape,
    Squeeze,
    ExpandDims,
    Sum,
    Neg,
    Sub,
    Div,
    Pow,
    Sqrt,
    Mean,
    Exp,
    Log,
    Flatten,
    Clip,
    Stack,
    GetItem,
    SetItem,
    Where,
    LogicalNot,
    Min,
    Max,
    ReduceMin,
    ReduceMax,
    Std,
    Gather,
    Scatter,
    Tile,
    Repeat,
    Roll,
    Flip,
    Pad,
    MaskedFill,
)

from .operation.mathematical_funcations import (
    ReLU,
    Sigmoid,
    Tanh,
    Softplus,
    ELU,
    LeakyReLU,
    Softmax,
    LogSoftmax,
)
from .operation.scientific_operations import (
    Dot,
    Outer,
    Norm,
    Det,
    Inv,
    Trace,
    Diag,
    SVD,
    Abs,
    Sign,
    Sin,
    Cos,
    Tan,
    Arcsin,
    Arccos,
    Arctan,
    Sinh,
    Cosh,
    Log2,
    Log10,
    Log1p,
    Expm1,
    Ceil,
    Floor,
    Round,
    Erf,
    Cumsum,
    Cumprod,
    ArgMax,
    ArgMin,
    TopK,
)


def _ensure_tensor(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def _ensure_numpy(data, dt):
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

    def __init__(self, data, dtype=None, requires_grad=False, creator=None):
        self.data = _ensure_numpy(data, dtype)

        self.requires_grad = requires_grad
        self.creator = creator
        self.grad = None
        self._version = 0
        self._base = None
        self._is_view = False
        self._id = Tid_count.get_incremented()

    def backward(self, grad=None, verbose=False):
        start = time.perf_counter()
        run_backward(self, grad)
        end = time.perf_counter()
        if verbose:
            print(f"Backward pass took {_format_time(end - start)}")

    def print(self):
        print(self.to_list())

    def zero_grad(self):
        self.grad = None

    def requires_grad_(self, flag=True):
        self.requires_grad = flag
        return self

    def astype(self, dtype):
        return Tensor(self.data.astype(dtype), requires_grad=self.requires_grad)

    def float(self):
        return self.astype(np.float32)

    def int(self):
        return self.astype(np.int32)

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

    def item(self):
        assert self.data.size == 1
        return self.data.item()

    def clone(self):
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def copy(self):
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def to_numpy(self):
        return self.data.copy()

    def tolist(self):
        return self.data.copy().tolist()

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    def __add__(self, other):
        other = _ensure_tensor(other)
        return Add.apply(self, other)

    def __mul__(self, other):
        other = _ensure_tensor(other)
        return Mul.apply(self, other)

    def __sub__(self, other):
        other = _ensure_tensor(other)
        return Sub.apply(self, other)

    def __truediv__(self, other):
        other = _ensure_tensor(other)
        return Div.apply(self, other)

    def __neg__(self):
        return Neg.apply(self)

    def __pow__(self, power):
        return Pow.apply(self, power=power)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __rpow__(self, other):
        return Tensor(other) ** self

    def __matmul__(self, other):
        other = _ensure_tensor(other)
        return MatMul.apply(self, other)

    def __invert__(self):
        return LogicalNot.apply(self)

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        other = _ensure_tensor(other)
        data = self.data == other.data
        return Tensor(data)

    def __gt__(self, other):
        other = _ensure_tensor(other)
        data = (self.data > other.data).astype(np.float32)
        return Tensor(data)

    def __lt__(self, other):
        other = _ensure_tensor(other)
        data = (self.data < other.data).astype(np.float32)
        return Tensor(data)

    def __ge__(self, other):
        other = _ensure_tensor(other)
        data = (self.data >= other.data).astype(np.float32)
        return Tensor(data)

    def __le__(self, other):
        other = _ensure_tensor(other)
        data = (self.data <= other.data).astype(np.float32)
        return Tensor(data)

    def __rgt__(self, other):
        other = _ensure_tensor(other)
        data = (other.data > self.data).astype(np.float32)
        return Tensor(data)

    def __rlt__(self, other):
        other = _ensure_tensor(other)
        data = (other.data < self.data).astype(np.float32)
        return Tensor(data)

    def __rge__(self, other):
        other = _ensure_tensor(other)
        data = (other.data >= self.data).astype(np.float32)
        return Tensor(data)

    def __rle__(self, other):
        other = _ensure_tensor(other)
        data = (other.data <= self.data).astype(np.float32)
        return Tensor(data)

    def __getitem__(self, idx):
        out = GetItem.apply(self, idx=idx)
        out._base = self
        out._is_view = True
        out._version = self._version
        return out

    def __setitem__(self, idx, value):
        value = _ensure_tensor(value)
        self._check_inplace_safe()
        self.data[idx] = value.data
        self._bump_version()

    # def __setitem__(self, idx, value):
    #         if not isinstance(value, Tensor):
    #             value = Tensor(value)
    #
    #         if self.requires_grad or value.requires_grad:
    #             new_tensor = SetItem.apply(self, value, idx=idx)
    #
    #             self.data = new_tensor.data
    #             self.creator = new_tensor.creator
    #             self.requires_grad = new_tensor.requires_grad
    #         else:
    #             self.data[idx] = value.data if isinstance(value, Tensor) else value
    def __array__(self, dtype=None):
        return self.data if dtype is None else self.data.astype(dtype)
    def __hash__(self):
     return id(self)
    def sum(self, axis=None, keepdims=False):
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def pow(self, power):
        return Pow.apply(self, power=power)

    def sqrt(self):
        return Sqrt.apply(self)

    def std(self, axis=None, keepdims=False):
        return Std.apply(self, axis=axis, keepdims=keepdims)

    def exp(self):
        return Exp.apply(self)

    def log(self):
        return Log.apply(self)

    def mean(self, axis=None, keepdims=False):
        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def min(self, other=None, axis=None):
        if other is None:
            return ReduceMin.apply(self, axis=axis)
        other = _ensure_tensor(other)
        return Min.apply(self, other)

    def max(self, other=None, axis=None):
        if other is None:
            return ReduceMax.apply(self, axis=axis)
        other = _ensure_tensor(other)
        return Max.apply(self, other)

    # mathematical_funcations

    def relu(self):
        return ReLU.apply(self)

    def sigmoid(self):
        return Sigmoid.apply(self)

    def tanh(self):
        return Tanh.apply(self)

    def softmax(self, axis=-1):
        return Softmax.apply(self, axis=axis)

    def log_softmax(self, axis=-1):
        return LogSoftmax.apply(self, axis=axis)

    def softplus(self):
        return Softplus.apply(self)

    def leakyrelu(self, alpha=0.01):
        return LeakyReLU.apply(self, alpha=alpha)

    def elu(self, alpha=0.01):
        return ELU.apply(self, alpha=alpha)

    # shape shiftier
    def reshape(self, *shape):
        return Reshape.apply(self, shape=shape)

    def clip(self, min_value, max_value):
        return Clip.apply(self, min_value=min_value, max_value=max_value)

    def transpose(self, axes=None):
        return Transpose.apply(self, axes=axes)

    @property
    def T(self):
        return Transpose.apply(self, axes=None)

    def clip_grad(self, max_norm):
        if self.grad is not None:
            norm = np.linalg.norm(self.grad)
            if norm > max_norm:
                self.grad *= max_norm / (norm + 1e-8)

    def zero_(self):
        self.data.fill(0)
        return self

    def permute(self, *axes):
        return self.transpose(axes)

    def squeeze(self, axis=None):
        return Squeeze.apply(self, axis=axis)

    def unsqueeze(self, axis):
        return ExpandDims.apply(self, axis=axis)

    def flatten(self):
        return Flatten.apply(self)

    def gather(self, axis, index):
        index = _ensure_tensor(index)
        return Gather.apply(self, index, axis=axis)

    def scatter(self, axis, index, shape):
        index = _ensure_tensor(index)
        return Scatter.apply(self, index, shape=shape, axis=axis)

    def tile(self, reps):
        return Tile.apply(self, reps=reps)

    def repeat(self, repeats, axis=None):
        return Repeat.apply(self, repeats=repeats, axis=axis)

    def roll(self, shift, axis=None):
        return Roll.apply(self, shift=shift, axis=axis)

    def flip(self, axis=None):
        return Flip.apply(self, axis=axis)

    def pad(self, pad_width, mode="constant", constant_value=0):
        return Pad.apply(
            self, pad_width=pad_width, mode=mode, constant_value=constant_value
        )

    def masked_fill(self, mask, fill_value):
        return MaskedFill.apply(self, mask=mask, fill_value=fill_value)

    def unbind(self, axis=0):
        n = self.shape[axis]
        return tuple(
            self[tuple(i if j == axis else slice(None) for j in range(self.ndim))]
            for i in range(n)
        )

    def topk(self, k, axis=-1, largest=True):
        return TopK.apply(self, k=k, axis=axis, largest=largest)

    def cumsum(self, axis=None):
        return Cumsum.apply(self, axis=axis)

    def cumprod(self, axis=None):
        return Cumprod.apply(self, axis=axis)

    def argmax(self, axis=None):
        return ArgMax.apply(self, axis=axis)

    def argmin(self, axis=None):
        return ArgMin.apply(self, axis=axis)

    # ── Linear Algebra ────────────────────────────────────────────────────────────

    def dot(self, other):
        return Dot.apply(self, _ensure_tensor(other))

    def outer(self, other):
        return Outer.apply(self, _ensure_tensor(other))

    def norm(self, ord=2, axis=None, keepdims=False):
        return Norm.apply(self, ord=ord, axis=axis, keepdims=keepdims)

    def det(self):
        return Det.apply(self)

    def inv(self):
        return Inv.apply(self)

    def trace(self):
        return Trace.apply(self)

    def diag(self):
        return Diag.apply(self)

    def svd(self):
        """Returns (U, S, Vt) each as a Tensor."""
        U_data, S_data, Vt_data = np.linalg.svd(self.data, full_matrices=False)
        # wrap each as leaf — gradient through SVD is rarely needed in practice
        return (Tensor(U_data), Tensor(S_data), Tensor(Vt_data))

    # ── Scientific ────────────────────────────────────────────────────────────────

    def abs(self):
        return Abs.apply(self)

    def sign(self):
        return Sign.apply(self)

    def sin(self):
        return Sin.apply(self)

    def cos(self):
        return Cos.apply(self)

    def tan(self):
        return Tan.apply(self)

    def arcsin(self):
        return Arcsin.apply(self)

    def arccos(self):
        return Arccos.apply(self)

    def arctan(self):
        return Arctan.apply(self)

    def sinh(self):
        return Sinh.apply(self)

    def cosh(self):
        return Cosh.apply(self)

    def log2(self):
        return Log2.apply(self)

    def log10(self):
        return Log10.apply(self)

    def log1p(self):
        return Log1p.apply(self)

    def expm1(self):
        return Expm1.apply(self)

    def ceil(self):
        return Ceil.apply(self)

    def floor(self):
        return Floor.apply(self)

    def round(self, decimals=0):
        # decimals != 0 is just a scale trick
        if decimals == 0:
            return Round.apply(self)
        scale = 10**decimals
        return (self * scale).round() / scale

    def erf(self):
        return Erf.apply(self)

    # ── Static factories ──────────────────────────────────────────────────────────

    # Utils

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

    # In-place operations

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

    def add_(self, other):
        other = _ensure_tensor(other)
        self.value = self.data + other.data
        self._bump_version()
        return self

    def sub_(self, other):
        other = _ensure_tensor(other)
        self.value = self.data - other.data
        self._bump_version()
        return self

    def mul_(self, other):
        other = _ensure_tensor(other)
        self.value = self.data * other.data
        self._bump_version()
        return self

    def div_(self, other):
        other = _ensure_tensor(other)
        self.value = self.data / other.data
        self._bump_version()
        return self

    # tensor display and visualization

    def __repr__(self):
        return print_tensor(self)

    def get_graph(self, max_depth=None, use_color=True):
        return get_graph(self, max_depth=max_depth, use_color=use_color)

    def show_graph(self, max_depth=None, use_color=True):
        print(get_graph(self, max_depth=max_depth, use_color=use_color))

    # -------- Creation --------
    @staticmethod
    def array(arr, dtype=None, requires_grad=False):
        return TensorFactory.array(arr, dtype, requires_grad)

    @staticmethod
    def tensor(arr, dtype=None, requires_grad=False):
        return TensorFactory.tensor(arr, dtype, requires_grad)

    @staticmethod
    def zeros(shape, dtype=None, requires_grad=False):
        return TensorFactory.zeros(shape, dtype, requires_grad)

    @staticmethod
    def ones(shape, dtype=None, requires_grad=False):
        return TensorFactory.ones(shape, dtype, requires_grad)

    @staticmethod
    def zeros_like(x, requires_grad=False):
        return TensorFactory.zeros_like(x, requires_grad)

    @staticmethod
    def ones_like(x, requires_grad=False):
        return TensorFactory.ones_like(x, requires_grad)

    # -------- Random --------
    @staticmethod
    def randn(shape, dtype=None, requires_grad=False):
        return TensorFactory.randn(shape, dtype, requires_grad)

    @staticmethod
    def rand(shape, dtype=None, requires_grad=False):
        return TensorFactory.rand(shape, dtype, requires_grad)

    @staticmethod
    def randint(low, high, shape, dtype=None, requires_grad=False):
        return TensorFactory.randint(low, high, shape, dtype, requires_grad)

    @staticmethod
    def uniform(low, high, shape, dtype=None, requires_grad=False):
        return TensorFactory.uniform(low, high, shape, dtype, requires_grad)

    # -------- Structured --------
    @staticmethod
    def arange(start, end=None, step=1, dtype=None, requires_grad=False):
        return TensorFactory.arange(start, end, step, dtype, requires_grad)

    @staticmethod
    def linspace(start, end, steps, dtype=None, requires_grad=False):
        return TensorFactory.linspace(start, end, steps, dtype, requires_grad)

    @staticmethod
    def eye(n, dtype=None, requires_grad=False):
        return TensorFactory.eye(n, dtype, requires_grad)

    @staticmethod
    def full(shape, fill_value, dtype=None, requires_grad=False):
        return TensorFactory.full(shape, fill_value, dtype, requires_grad)

    @staticmethod
    def full_like(x, fill_value, requires_grad=False):
        return TensorFactory.full_like(x, fill_value, requires_grad)

    # -------- Combine --------
    @staticmethod
    def concat(tensors, axis=0):
        return TensorFactory.concat(tensors, axis)

    @staticmethod
    def stack(tensors, axis=0):
        return TensorFactory.stack(tensors, axis)

    @staticmethod
    def meshgrid(*tensors, indexing="xy"):
        grids = np.meshgrid(*[t.data for t in tensors], indexing=indexing)
        return tuple(Tensor(g) for g in grids)

    @staticmethod
    def tril(x, diagonal=0):
        t = _ensure_tensor(x)
        return Tensor(np.tril(t.data, k=diagonal))

    @staticmethod
    def triu(x, diagonal=0):
        t = _ensure_tensor(x)
        return Tensor(np.triu(t.data, k=diagonal))

    # -------- Elementwise --------
    @staticmethod
    def where(cond, x, y):
        return TensorFactory.where(cond, x, y)

    @staticmethod
    def minimum(x, y):
        return TensorFactory.minimum(x, y)

    @staticmethod
    def maximum(x, y):
        return TensorFactory.maximum(x, y)

    @staticmethod
    def checkpoint(fn, args):
        if isinstance(args, tuple):
            return Checkpoint.apply(fn, *args)
        return Checkpoint.apply(fn, args)

    # -------- Utils --------
    @staticmethod
    def export_graph(tensor, file, as_image=False):
        return export_graph(tensor, file, as_image)

    @staticmethod
    def seed(seed):
        return TensorFactory.seed(seed)

    @staticmethod
    def stop_global_grad():
        Global_grad.stop_global_grad()

    @staticmethod
    def release_global_grad():
        Global_grad.release_global_grad()

    @staticmethod
    def no_grad():
        return No_grad()


TensorFactory.Tensor = Tensor
