import numpy as np


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


def ensure_tensor(x):
    from .tensor import Tensor

    if isinstance(x, Tensor):
        return x
    return Tensor(x)


# Arithmetic


def add(x, y):
    return Add.apply(ensure_tensor(x), ensure_tensor(y))


def mul(x, y):
    return Mul.apply(ensure_tensor(x), ensure_tensor(y))


def sub(x, y):
    return Sub.apply(ensure_tensor(x), ensure_tensor(y))


def div(x, y):
    return Div.apply(ensure_tensor(x), ensure_tensor(y))


def neg(x):
    return Neg.apply(ensure_tensor(x))


def pow(x, power):
    return Pow.apply(ensure_tensor(x), power=power)


def sqrt(x):
    return Sqrt.apply(ensure_tensor(x))


# Reductions


def sum(x, axis=None, keepdims=False):
    return Sum.apply(ensure_tensor(x), axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    return Mean.apply(ensure_tensor(x), axis=axis, keepdims=keepdims)


def std(x, axis=None, keepdims=False):
    return Std.apply(ensure_tensor(x), axis=axis, keepdims=keepdims)


def min(x, y=None, axis=None):
    x = ensure_tensor(x)

    if y is None:
        return ReduceMin.apply(x, axis=axis)

    return Min.apply(x, ensure_tensor(y))


def max(x, y=None, axis=None):
    x = ensure_tensor(x)

    if y is None:
        return ReduceMax.apply(x, axis=axis)

    return Max.apply(x, ensure_tensor(y))


# Shape Operations


def reshape(x, *shape):
    return Reshape.apply(ensure_tensor(x), shape=shape)


def transpose(x, axes=None):
    return Transpose.apply(ensure_tensor(x), axes=axes)


def permute(x, *axes):
    return transpose(x, axes)


def squeeze(x, axis=None):
    return Squeeze.apply(ensure_tensor(x), axis=axis)


def unsqueeze(x, axis):
    return ExpandDims.apply(ensure_tensor(x), axis=axis)


def flatten(x):
    return Flatten.apply(ensure_tensor(x))


def clip(x, min_value, max_value):
    return Clip.apply(ensure_tensor(x), min_value=min_value, max_value=max_value)


# Tensor Combine


def concat(tensors, axis=0):
    tensors = [ensure_tensor(t) for t in tensors]
    return Concat.apply(*tensors, axis=axis)


def stack(tensors, axis=0):
    tensors = [ensure_tensor(t) for t in tensors]
    return Stack.apply(*tensors, axis=axis)


# Matrix / Linear Algebra


def matmul(x, y):
    return MatMul.apply(ensure_tensor(x), ensure_tensor(y))


def dot(x, y):
    return Dot.apply(ensure_tensor(x), ensure_tensor(y))


def outer(x, y):
    return Outer.apply(ensure_tensor(x), ensure_tensor(y))


def norm(x, ord=2, axis=None, keepdims=False):
    return Norm.apply(ensure_tensor(x), ord=ord, axis=axis, keepdims=keepdims)


def det(x):
    return Det.apply(ensure_tensor(x))


def inv(x):
    return Inv.apply(ensure_tensor(x))


def trace(x):
    return Trace.apply(ensure_tensor(x))


def diag(x):
    return Diag.apply(ensure_tensor(x))


def svd(x):
    return SVD.apply(ensure_tensor(x))


# Activations


def relu(x):
    return ReLU.apply(ensure_tensor(x))


def sigmoid(x):
    return Sigmoid.apply(ensure_tensor(x))


def tanh(x):
    return Tanh.apply(ensure_tensor(x))


def softplus(x):
    return Softplus.apply(ensure_tensor(x))


def elu(x, alpha=0.01):
    return ELU.apply(ensure_tensor(x), alpha=alpha)


def leakyrelu(x, alpha=0.01):
    return LeakyReLU.apply(ensure_tensor(x), alpha=alpha)


def softmax(x, axis=-1):
    return Softmax.apply(ensure_tensor(x), axis=axis)


def log_softmax(x, axis=-1):
    return LogSoftmax.apply(ensure_tensor(x), axis=axis)


# Scientific


def abs(x):
    return Abs.apply(ensure_tensor(x))


def sign(x):
    return Sign.apply(ensure_tensor(x))


def sin(x):
    return Sin.apply(ensure_tensor(x))


def cos(x):
    return Cos.apply(ensure_tensor(x))


def tan(x):
    return Tan.apply(ensure_tensor(x))


def arcsin(x):
    return Arcsin.apply(ensure_tensor(x))


def arccos(x):
    return Arccos.apply(ensure_tensor(x))


def arctan(x):
    return Arctan.apply(ensure_tensor(x))


def sinh(x):
    return Sinh.apply(ensure_tensor(x))


def cosh(x):
    return Cosh.apply(ensure_tensor(x))


def exp(x):
    return Exp.apply(ensure_tensor(x))


def log(x):
    return Log.apply(ensure_tensor(x))


def log2(x):
    return Log2.apply(ensure_tensor(x))


def log10(x):
    return Log10.apply(ensure_tensor(x))


def log1p(x):
    return Log1p.apply(ensure_tensor(x))


def expm1(x):
    return Expm1.apply(ensure_tensor(x))


def ceil(x):
    return Ceil.apply(ensure_tensor(x))


def floor(x):
    return Floor.apply(ensure_tensor(x))


def round(x, decimals=0):
    x = ensure_tensor(x)

    if decimals == 0:
        return Round.apply(x)

    scale = 10**decimals
    return Round.apply(x * scale) / scale


def erf(x):
    return Erf.apply(ensure_tensor(x))


# Indexing


def gather(x, axis, index):
    return Gather.apply(ensure_tensor(x), ensure_tensor(index), axis=axis)


def scatter(x, axis, index, shape):
    return Scatter.apply(ensure_tensor(x), ensure_tensor(index), shape=shape, axis=axis)


def where(cond, x, y):
    return Where.apply(ensure_tensor(x), ensure_tensor(y), cond=cond)


def logical_not(x):
    return LogicalNot.apply(ensure_tensor(x))


# Tensor Manipulation


def tile(x, reps):
    return Tile.apply(ensure_tensor(x), reps=reps)


def repeat(x, repeats, axis=None):
    return Repeat.apply(ensure_tensor(x), repeats=repeats, axis=axis)


def roll(x, shift, axis=None):
    return Roll.apply(ensure_tensor(x), shift=shift, axis=axis)


def flip(x, axis=None):
    return Flip.apply(ensure_tensor(x), axis=axis)


def pad(x, pad_width, mode="constant", constant_value=0):
    return Pad.apply(
        ensure_tensor(x), pad_width=pad_width, mode=mode, constant_value=constant_value
    )


def masked_fill(x, mask, fill_value):
    return MaskedFill.apply(ensure_tensor(x), mask=mask, fill_value=fill_value)


# Statistics


def cumsum(x, axis=None):
    return Cumsum.apply(ensure_tensor(x), axis=axis)


def cumprod(x, axis=None):
    return Cumprod.apply(ensure_tensor(x), axis=axis)


def argmax(x, axis=None):
    return ArgMax.apply(ensure_tensor(x), axis=axis)


def argmin(x, axis=None):
    return ArgMin.apply(ensure_tensor(x), axis=axis)


def topk(x, k, axis=-1, largest=True):
    return TopK.apply(ensure_tensor(x), k=k, axis=axis, largest=largest)


# Special


def getitem(x, idx):
    return GetItem.apply(ensure_tensor(x), idx=idx)


def setitem(x, idx, value):
    return SetItem.apply(ensure_tensor(x), ensure_tensor(value), idx=idx)



# Exports


__all__ = [name for name in globals() if not name.startswith("_")]
