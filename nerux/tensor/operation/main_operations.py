import numpy as np
from ..function import Function
from ..utils import reduce_grad


def ensure_array(data, dtype=None):
    if isinstance(data, np.ndarray):
        return data.astype(dtype) if dtype else data

    return np.array(data, dtype=dtype if dtype else np.float32)


# ========== Basic Math Ops ==========


class Add(Function):
    def forward(self, a, b):
        a, b = ensure_array(a), ensure_array(b)
        self.a_shape, self.b_shape = a.shape, b.shape
        return a + b

    def backward(self, grad_output):
        ga = reduce_grad(grad_output, self.a_shape)
        gb = reduce_grad(grad_output, self.b_shape)
        return ga, gb


class Sub(Function):
    def forward(self, a, b):
        a, b = ensure_array(a), ensure_array(b)
        self.a_shape, self.b_shape = a.shape, b.shape
        return a - b

    def backward(self, grad_output):
        ga = reduce_grad(grad_output, self.a_shape)
        gb = reduce_grad(-grad_output, self.b_shape)
        return ga, gb


class Mul(Function):
    def forward(self, a, b):
        a, b = ensure_array(a), ensure_array(b)
        self.a, self.b = a, b
        return a * b

    def backward(self, grad_output):
        ga = grad_output * self.b
        gb = grad_output * self.a
        return reduce_grad(ga, self.a.shape), reduce_grad(gb, self.b.shape)


class LogicalNot(Function):
    def forward(self, a):
        self.a = a
        return np.logical_not(a)

    def backward(self, grad_output):
        return np.zeros_like(self.a)


class Div(Function):
    def forward(self, a, b):
        self.a, self.b = a, b
        return a / b

    def backward(self, grad_output):
        ga = grad_output / self.b
        gb = -grad_output * self.a / (self.b**2)
        return reduce_grad(ga, self.a.shape), reduce_grad(gb, self.b.shape)


class MatMul2(Function):
    def forward(self, a, b):
        a, b = ensure_array(a), ensure_array(b)
        self.a, self.b = a, b
        return a @ b

    def backward(self, grad_output):
        a, b, g = self.a, self.b, grad_output

        # -------- Case 1: 2D @ 2D --------
        if a.ndim == 2 and b.ndim == 2:
            grad_a = g @ b.T
            grad_b = a.T @ g
            return grad_a, grad_b

        # -------- Case 2: Batched @ Batched --------
        elif a.ndim >= 3 and b.ndim >= 3:
            b_T = np.swapaxes(b, -1, -2)
            a_T = np.swapaxes(a, -1, -2)

            grad_a = g @ b_T
            grad_b = a_T @ g
            return grad_a, grad_b

        # -------- Case 3: (B,L,D) @ (D,V) --------
        elif a.ndim >= 3 and b.ndim == 2:
            # grad_a: same shape as a
            grad_a = g @ b.T

            # grad_b: collapse batch + seq
            a_flat = a.reshape(-1, a.shape[-1])  # (B*L, D)
            g_flat = g.reshape(-1, g.shape[-1])  # (B*L, V)

            grad_b = a_flat.T @ g_flat  # (D, V)

            return grad_a, grad_b

        # -------- Case 4: (D,V) @ (B,L,V) -------- (rare but for completeness)

        elif a.ndim == 2 and b.ndim >= 3:
            # treat a as (1, D, V) broadcast
            b_T = np.swapaxes(b, -1, -2)  # (B, V, L)
            grad_a = (g @ b_T).sum(axis=0)  # (D, V) after summing batch
            a_T = a.T  # (V, D)
            # grad_b = a_T @ g  # (B, V, L) — wrong direction
            # correct:
            grad_b = np.einsum("ij,bkj->bki", a, g)  # cleanest for this case
            return grad_a, grad_b

        else:
            raise NotImplementedError(
                f"Unsupported matmul shapes: {a.shape} @ {b.shape}"
            )


def unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class MatMul(Function):
    def forward(self, a, b):
        a, b = ensure_array(a), ensure_array(b)

        self.a_shape = a.shape
        self.b_shape = b.shape

        self.a_was_1d = a.ndim == 1
        self.b_was_1d = b.ndim == 1

        # Promote 1D tensors
        if self.a_was_1d:
            a = a[None, :]  # (D,) -> (1,D)

        if self.b_was_1d:
            b = b[:, None]  # (D,) -> (D,1)

        self.a = a
        self.b = b

        out = a @ b

        # Remove temporary dims
        if self.a_was_1d:
            out = out.squeeze(-2)

        if self.b_was_1d:
            out = out.squeeze(-1)

        return out

    def backward(self, grad_output):
        g = grad_output
        a = self.a
        b = self.b

        # Re-expand grad dimensions
        if self.a_was_1d:
            g = np.expand_dims(g, axis=-2)

        if self.b_was_1d:
            g = np.expand_dims(g, axis=-1)

        b_T = np.swapaxes(b, -1, -2)
        a_T = np.swapaxes(a, -1, -2)

        grad_a = g @ b_T
        grad_b = a_T @ g

        # Remove fake dims
        if self.a_was_1d:
            grad_a = grad_a.squeeze(-2)

        if self.b_was_1d:
            grad_b = grad_b.squeeze(-1)

        # Handle broadcasted batch dims
        grad_a = unbroadcast(grad_a, self.a_shape)
        grad_b = unbroadcast(grad_b, self.b_shape)

        return grad_a, grad_b


# ========== Reduction Ops ==========


class Sum(Function):
    def __init__(self, *inputs, axis=None, keepdims=False):
        super().__init__(*inputs)
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, a):
        self.input_shape = a.shape

        return np.array(a.sum(axis=self.axis, keepdims=self.keepdims), dtype=float)

    def backward(self, grad_output):
        grad = grad_output

        if not self.keepdims and self.axis is not None:
            grad = np.expand_dims(grad, axis=self.axis)

        return np.ones(self.input_shape, dtype=float) * grad


class Mean(Function):
    def __init__(self, *inputs, axis=None, keepdims=False):
        super().__init__(*inputs)
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, a):
        self.input_shape = a.shape

        if self.axis is None:
            self.count = np.prod(a.shape)
        else:
            if isinstance(self.axis, int):
                axes = [self.axis]
            else:
                axes = self.axis
            self.count = np.prod([a.shape[ax] for ax in axes])

        out = np.array(a.mean(axis=self.axis, keepdims=self.keepdims), dtype=float)
        return out

    def backward(self, grad_output):
        grad = grad_output / self.count

        if not self.keepdims and self.axis is not None:
            grad = np.expand_dims(grad, axis=self.axis)

        return np.ones(self.input_shape, dtype=float) * grad


# ========== Unary Ops ==========


class Neg(Function):
    def forward(self, a):
        return -a

    def backward(self, grad_output):
        return -grad_output


class Pow(Function):
    def __init__(self, *inputs, power=2):
        super().__init__(*inputs)
        self.power = power

    def forward(self, a):
        self.saved_a = a
        return np.power(a, self.power)

    def backward(self, grad_output):
        grad_a = grad_output * self.power * np.power(self.saved_a, self.power - 1)
        return grad_a


class Sqrt(Function):
    def forward(self, a):
        self.saved_a = a
        return np.sqrt(a)

    def backward(self, grad_output):
        grad_a = grad_output * (0.5 / np.sqrt(self.saved_a))
        return grad_a


class Std(Function):
    def __init__(self, x, axis=None, keepdims=False, eps=1e-8):
        super().__init__(x)
        self.axis = axis
        self.keepdims = keepdims
        self.eps = eps

    def forward(self, x):
        self.x = x

        self.mean = np.mean(x, axis=self.axis, keepdims=True)
        self.var = np.mean((x - self.mean) ** 2, axis=self.axis, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)

        if self.keepdims:
            return self.std
        else:
            return np.squeeze(self.std, axis=self.axis)

    def backward(self, grad_output):
        x = self.x
        mean = self.mean
        std = self.std

        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, axis=self.axis)

        if self.axis is None:
            N = x.size
        else:
            axes = self.axis if isinstance(self.axis, tuple) else (self.axis,)
            N = np.prod([x.shape[a] for a in axes])

        grad_x = grad_output * (x - mean) / (N * std)

        return grad_x


class Exp(Function):
    def forward(self, a):
        out = np.exp(a)
        self.saved_out = out
        return out

    def backward(self, grad_output):
        return grad_output * self.saved_out


class Log(Function):
    def forward(self, a):
        self.saved_a = a
        return np.log(a)

    def backward(self, grad_output):
        return grad_output / self.saved_a


# ========== MinMax Ops ==========


class Max(Function):
    def forward(self, x, y):
        self.x_mask = x >= y
        self.y_mask = ~self.x_mask
        out = np.where(self.x_mask, x, y)

        tie = x == y
        self.x_mask = self.x_mask & ~tie
        self.y_mask = self.y_mask & ~tie
        self.tie = tie
        return out

    def backward(self, grad_output):
        gx = grad_output * (self.x_mask.astype(float) + self.tie * 0.5)
        gy = grad_output * (self.y_mask.astype(float) + self.tie * 0.5)
        return gx, gy


class Min(Function):
    def forward(self, x, y):
        self.x_mask = x <= y
        self.y_mask = ~self.x_mask
        out = np.where(self.x_mask, x, y)

        tie = x == y
        self.x_mask = self.x_mask & ~tie
        self.y_mask = self.y_mask & ~tie
        self.tie = tie
        return out

    def backward(self, grad_output):
        gx = grad_output * (self.x_mask.astype(float) + self.tie * 0.5)
        gy = grad_output * (self.y_mask.astype(float) + self.tie * 0.5)
        return gx, gy


# class Min(Function):
#     def forward(self, x, y):
#         self.mask = x < y
#         return np.where(self.mask, x, y)
#
#     def backward(self, grad_output):
#         gx = grad_output * self.mask
#         gy = grad_output * (~self.mask)
#         return gx, gy
#
#
# class Max(Function):
#     def forward(self, x, y):
#         self.mask = x > y
#         return np.where(self.mask, x, y)
#
#     def backward(self, grad_output):
#         gx = grad_output * self.mask
#         gy = grad_output * (~self.mask)
#         return gx, gy
#


class ReduceMin(Function):
    def __init__(self, x, axis=None):
        super().__init__(x)
        self.axis = axis

    def forward(self, x):
        self.x_shape = x.shape
        min_val = x.min(axis=self.axis, keepdims=True)
        self.mask = (x == min_val).astype(float)
        # normalise so ties split the gradient evenly
        self.mask /= self.mask.sum(axis=self.axis, keepdims=True)
        return x.min(axis=self.axis)  # return without keepdims for correct output shape

    def backward(self, grad_output):
        if self.axis is not None:
            grad_output = np.expand_dims(grad_output, axis=self.axis)
        return grad_output * self.mask


class ReduceMax(Function):
    def __init__(self, x, axis=None):
        super().__init__(x)
        self.axis = axis

    def forward(self, x):
        self.x_shape = x.shape
        min_val = x.max(axis=self.axis, keepdims=True)
        self.mask = (x == min_val).astype(float)
        # normalise so ties split the gradient evenly
        self.mask /= self.mask.sum(axis=self.axis, keepdims=True)
        return x.max(axis=self.axis)  # return without keepdims for correct output shape

    def backward(self, grad_output):
        if self.axis is not None:
            grad_output = np.expand_dims(grad_output, axis=self.axis)
        return grad_output * self.mask


# class ReduceMin(Function):
#     def __init__(self, x, axis=None):
#         super().__init__(x)
#         self.axis = axis
#
#     def forward(self, x):
#         self.x = x
#         self.min_val = x.min(axis=self.axis)
##        store mask of argmin positions
#         self.mask = x == self.min_val
#         return self.min_val
#
#     def backward(self, grad_output):
# &       grad_output is scalar → expand to mask
#         return grad_output * self.mask
#
#
# class ReduceMax(Function):
#     def __init__(self, x, axis=None):
#         super().__init__(x)
#         self.axis = axis
#
#     def forward(self, x):
#         self.x = x
#         self.max_val = x.max(axis=self.axis)
#         self.mask = x == self.max_val
#         return self.max_val
#
#     def backward(self, grad_output):
#         return grad_output * self.mask


# ========== Shape Ops ==========


class GetItem(Function):
    """
    Indexing operation: x[idx]
    Supports: integers, slices, tuples of indices/slices, boolean masks
    """

    def __init__(self, *inputs, idx):
        super().__init__(*inputs)
        self.idx = idx

    def forward(self, x):
        self.x_shape = x.shape
        return x[self.idx]

    def backward(self, grad_output):
        # Create gradient with same shape as input, filled with zeros
        grad_x = np.zeros(self.x_shape)

        # Add gradients only to the indexed positions
        # grad_x[self.idx] = grad_output
        np.add.at(grad_x, self.idx, grad_output)

        return grad_x


class SetItem(Function):
    """
    In-place assignment: x[idx] = value
    This is tricky for autograd - we create a new tensor with updated values
    """

    def __init__(self, *inputs, idx):
        super().__init__(*inputs)
        self.idx = idx

    def forward(self, x, value):
        self.x_shape = x.shape
        self.value_shape = value.shape if hasattr(value, "shape") else ()

        # Create a copy to avoid modifying original
        result = x.copy()
        result[self.idx] = value
        return result

    def backward(self, grad_output):
        # Gradient w.r.t x: same as grad_output but zero at indexed positions
        grad_x = grad_output.copy()
        grad_x[self.idx] = 0

        # Gradient w.r.t value: grad at indexed positions
        grad_value = grad_output[self.idx]

        return grad_x, grad_value


class Flatten(Function):
    def forward(self, a):
        self.original_shape = a.shape
        # a.reshape(a.shape[0], -1)
        return a.flatten()

    def backward(self, grad_output):
        return grad_output.reshape(self.original_shape)


class Clip(Function):
    def __init__(self, *inputs, min_value, max_value):
        super().__init__(*inputs)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, a):
        self.mask = (a >= self.min_value) & (a <= self.max_value)
        return np.clip(a, self.min_value, self.max_value)

    def backward(self, grad_output):
        return grad_output * self.mask


class ExpandDims(Function):
    def __init__(self, *inputs, axis):
        super().__init__(*inputs)
        self.axis = axis

    def forward(self, a):
        return np.expand_dims(a, axis=self.axis)

    def backward(self, grad_output):
        return np.squeeze(grad_output, axis=self.axis)


class Squeeze(Function):
    def __init__(self, *inputs, axis=None):
        super().__init__(*inputs)
        self.axis = axis

    def forward(self, a):
        self.input_shape = a.shape
        return np.squeeze(a, axis=self.axis)

    def backward(self, grad_output):
        return np.reshape(grad_output, self.input_shape)


class Reshape(Function):
    def __init__(self, *inputs, shape=None):
        super().__init__(*inputs)
        self.new_shape = shape

    def forward(self, a):
        self.original_shape = a.shape
        return np.reshape(a, self.new_shape)

    def backward(self, grad_output):
        return grad_output.reshape(self.original_shape)


class Transpose(Function):
    def __init__(self, *inputs, axes=None):
        super().__init__(*inputs)
        self.axes = axes

    def forward(self, a, *args):
        self.input_shape = a.shape
        if self.axes is None:
            # Default: reverse all axes
            return np.transpose(a)
        return np.transpose(a, axes=self.axes)

    def backward(self, grad_output):
        if self.axes is None:
            # Reverse transpose
            return np.transpose(grad_output)

        # Compute inverse permutation
        inv_axes = np.argsort(self.axes)
        return np.transpose(grad_output, axes=tuple(inv_axes))


class Gather(Function):
    """x.gather(axis, index) — like torch.gather"""

    def __init__(self, *inputs, axis):
        super().__init__(*inputs)
        self.axis = axis

    def forward(self, x, index):
        self.index = index.astype(int)
        self.x_shape = x.shape
        return np.take_along_axis(x, self.index, axis=self.axis)

    def backward(self, grad_output):
        grad_x = np.zeros(self.x_shape)
        np.add.at(
            grad_x,
            tuple(
                self.index
                if i == self.axis
                else np.arange(self.x_shape[i]).reshape(
                    [-1 if j == i else 1 for j in range(grad_x.ndim)]
                )
                for i in range(grad_x.ndim)
            ),
            grad_output,
        )
        return grad_x, None


class Scatter(Function):
    """
    Out-of-place scatter: place values into a zero tensor at given indices.
    result = zeros(shape); result[index] = src
    """

    def __init__(self, *inputs, shape, axis):
        super().__init__(*inputs)
        self.shape = shape
        self.axis = axis

    def forward(self, src, index):
        self.index = index.astype(int)
        self.src_shape = src.shape
        out = np.zeros(self.shape, dtype=src.dtype)
        np.add.at(
            out,
            tuple(
                self.index if i == self.axis else slice(None)
                for i in range(len(self.shape))
            ),
            src,
        )
        return out

    def backward(self, grad_output):
        grad_src = np.take_along_axis(grad_output, self.index, axis=self.axis)
        return grad_src, None


class Tile(Function):
    """np.tile — repeat tensor along axes"""

    def __init__(self, *inputs, reps):
        super().__init__(*inputs)
        self.reps = reps

    def forward(self, x):
        self.x_shape = x.shape
        return np.tile(x, self.reps)

    def backward(self, grad_output):
        # fold repeated dims back and sum
        grad = grad_output
        reps = self.reps
        x_shape = self.x_shape

        # pad reps to match ndim
        if len(reps) < grad.ndim:
            reps = (1,) * (grad.ndim - len(reps)) + tuple(reps)

        for axis, rep in enumerate(reps):
            if rep > 1:
                grad = grad.reshape(
                    *grad.shape[:axis],
                    rep,
                    grad.shape[axis] // rep,
                    *grad.shape[axis + 1 :],
                ).sum(axis=axis)
        return grad.reshape(x_shape)


class Repeat(Function):
    """np.repeat — repeat each element n times"""

    def __init__(self, *inputs, repeats, axis=None):
        super().__init__(*inputs)
        self.repeats = repeats
        self.axis = axis

    def forward(self, x):
        self.x_shape = x.shape
        return np.repeat(x, self.repeats, axis=self.axis)

    def backward(self, grad_output):
        if self.axis is None:
            grad_output = grad_output.reshape(-1)
            grad = grad_output.reshape(-1, self.repeats).sum(axis=1)
            return grad.reshape(self.x_shape)
        else:
            n = self.x_shape[self.axis]
            chunks = np.split(
                grad_output, n * self.repeats // self.repeats, axis=self.axis
            )
            return np.stack(
                [
                    sum(chunks[i * self.repeats : (i + 1) * self.repeats])
                    for i in range(n)
                ],
                axis=self.axis,
            )


class Roll(Function):
    """np.roll — circular shift"""

    def __init__(self, *inputs, shift, axis=None):
        super().__init__(*inputs)
        self.shift = shift
        self.axis = axis

    def forward(self, x):
        return np.roll(x, self.shift, axis=self.axis)

    def backward(self, grad_output):
        # reverse the roll
        return np.roll(grad_output, -self.shift, axis=self.axis)


class Flip(Function):
    """Reverse along given axes"""

    def __init__(self, *inputs, axis=None):
        super().__init__(*inputs)
        self.axis = axis

    def forward(self, x):
        return np.flip(x, axis=self.axis).copy()

    def backward(self, grad_output):
        return np.flip(grad_output, axis=self.axis).copy()


class Pad(Function):
    """
    Zero-pad a tensor.
    pad_width: same format as np.pad — ((before_0, after_0), (before_1, after_1), ...)
    """

    def __init__(self, *inputs, pad_width, mode="constant", constant_value=0):
        super().__init__(*inputs)
        self.pad_width = pad_width
        self.mode = mode
        self.constant_value = constant_value

    def forward(self, x):
        self.x_shape = x.shape
        return np.pad(
            x, self.pad_width, mode=self.mode, constant_values=self.constant_value
        )

    def backward(self, grad_output):
        # slice out the original region
        slices = tuple(
            slice(p[0], grad_output.shape[i] - p[1])
            for i, p in enumerate(self.pad_width)
        )
        return grad_output[slices]


class MaskedFill(Function):
    """
    Set positions where mask==True to fill_value.
    Out-of-place (returns new tensor).
    """

    def __init__(self, *inputs, mask, fill_value):
        super().__init__(*inputs)
        self.mask = (
            mask.data.astype(bool) if hasattr(mask, "data") else np.array(mask, bool)
        )
        self.fill_value = fill_value

    def forward(self, x):
        out = x.copy()
        out[self.mask] = self.fill_value
        return out

    def backward(self, grad_output):
        grad = grad_output.copy()
        grad[self.mask] = 0.0  # no gradient flows through filled positions
        return grad


class Concat(Function):
    def __init__(self, *arrays, axis=0):
        super().__init__(*arrays)
        self.axis = axis

    def forward(self, *arrays):
        self.shapes = [a.shape for a in arrays]
        return np.concatenate(arrays, axis=self.axis)

    def backward(self, grad_output):
        splits = np.split(
            grad_output,
            np.cumsum([s[self.axis] for s in self.shapes[:-1]]),
            axis=self.axis,
        )
        return tuple(splits)


class Stack(Function):
    def __init__(self, *tensors, axis=0):
        super().__init__(*tensors)
        self.axis = axis

    def forward(self, *tensors):
        self.num_inputs = len(tensors)
        self.axis = self.axis % (tensors[0].ndim + 1)
        self.input_shapes = [t.shape for t in tensors]
        return np.stack(tensors, axis=self.axis)

    def backward(self, grad_output):
        grads = np.split(grad_output, self.num_inputs, axis=self.axis)
        # Remove the stacking dimension (to match input shapes)
        grads = [g.squeeze(self.axis) for g in grads]
        return tuple(grads)


class Where(Function):
    def __init__(self, x, y, cond):
        super().__init__(x, y)
        self.cond = cond  # Tensor mask

    def forward(self, x, y):
        cond = self.cond
        self.mask = (cond.data if hasattr(cond, "data") else cond).astype(bool)
        return np.where(self.mask, x, y)

    def backward(self, grad_output):
        mask = self.mask

        gx = grad_output * mask
        gy = grad_output * (~mask)

        return gx, gy  # only two grads (for x and y)
