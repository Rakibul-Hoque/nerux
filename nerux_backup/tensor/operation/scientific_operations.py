import numpy as np
from ..function import Function
from ..utils import reduce_grad


def ensure_array(data, dtype=None):
    if isinstance(data, np.ndarray):
        return data.astype(dtype) if dtype else data

    return np.array(data, dtype=dtype if dtype else np.float32)


# ========== Linear Algebra ==========


class Dot(Function):
    """1-D dot product"""

    def forward(self, a, b):
        self.a, self.b = a, b
        return np.array(np.dot(a, b))

    def backward(self, grad_output):
        return grad_output * self.b, grad_output * self.a


class Outer(Function):
    """Outer product of two 1-D vectors → (N, M)"""

    def forward(self, a, b):
        self.a, self.b = a, b
        return np.outer(a, b)

    def backward(self, grad_output):
        grad_a = grad_output @ self.b
        grad_b = grad_output.T @ self.a
        return grad_a, grad_b


class Norm(Function):
    """
    L-p norm along an axis.
    ord=2 → Frobenius/L2, ord=1 → L1, etc.
    """

    def __init__(self, *inputs, ord=2, axis=None, keepdims=False):
        super().__init__(*inputs)
        self.ord = ord
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x = x
        out = np.linalg.norm(x, ord=self.ord, axis=self.axis, keepdims=True)
        self.norm_keepdims = out  # save for backward
        if not self.keepdims:
            out = np.squeeze(out, axis=self.axis)
        return out

    def backward(self, grad_output):
        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, axis=self.axis)

        if self.ord == 2 or self.ord is None:
            # d/dx ||x||_2 = x / ||x||_2
            return grad_output * self.x / (self.norm_keepdims + 1e-12)
        elif self.ord == 1:
            return grad_output * np.sign(self.x)
        else:
            # general: d/dx ||x||_p = x * |x|^(p-2) / ||x||_p^(p-1)
            p = self.ord
            abs_x_pm2 = np.abs(self.x) ** (p - 2)
            return (
                grad_output
                * self.x
                * abs_x_pm2
                / (self.norm_keepdims ** (p - 1) + 1e-12)
            )


class Det(Function):
    """Matrix determinant — d/dA det(A) = det(A) * A^{-T}"""

    def forward(self, x):
        self.x = x
        self.det = np.linalg.det(x)
        return np.array(self.det)

    def backward(self, grad_output):
        inv_T = np.linalg.inv(self.x).swapaxes(-1, -2)
        return grad_output * self.det[..., None, None] * inv_T


class Inv(Function):
    """Matrix inverse — d/dA A^{-1} = -A^{-T} dA A^{-T}"""

    def forward(self, x):
        self.inv = np.linalg.inv(x)
        return self.inv

    def backward(self, grad_output):
        inv_T = self.inv.swapaxes(-1, -2)
        return -inv_T @ grad_output @ inv_T


class Trace(Function):
    """Sum of diagonal elements"""

    def forward(self, x):
        self.x_shape = x.shape
        return np.array(np.trace(x))

    def backward(self, grad_output):
        return grad_output * np.eye(self.x_shape[-2], self.x_shape[-1])


class Diag(Function):
    """
    If input is 2D: extract diagonal → 1D.
    If input is 1D: create diagonal matrix → 2D.
    """

    def forward(self, x):
        self.x_shape = x.shape
        return np.diag(x)

    def backward(self, grad_output):
        if len(self.x_shape) == 1:
            return np.diag(grad_output)
        else:
            return np.diag(grad_output)


class SVD(Function):
    """
    Thin SVD: A = U S V^T
    Returns tuple (U, S, V) — each wrapped as Tensor in the method.
    Backward via custom VJP.
    """

    def forward(self, x):
        self.U, self.S, self.Vt = np.linalg.svd(x, full_matrices=False)
        # pack all three into one flat array so Function gets one output
        # actual unpacking happens in Tensor.svd()
        self._packed = (self.U, self.S, self.Vt)
        return self.U  # placeholder — svd handled specially in Tensor

    def backward(self, grad_output):
        # Analytical SVD VJP (Ionescu et al.)
        U, S, Vt = self.U, self.S, self.Vt
        V = Vt.T
        dU = grad_output

        F = 1.0 / (S[..., :, None] ** 2 - S[..., None, :] ** 2 + 1e-12)
        np.fill_diagonal(F, 0)

        UdU = U.T @ dU
        grad_A = U @ (F * (UdU - UdU.T)) @ Vt
        return grad_A


# ========== Scientific / Elementwise ==========


class Abs(Function):
    def forward(self, x):
        self.x = x
        return np.abs(x)

    def backward(self, grad_output):
        return grad_output * np.sign(self.x)


class Sign(Function):
    """d/dx sign(x) = 0 everywhere (subgradient: 0 at 0)"""

    def forward(self, x):
        self.x_shape = x.shape
        return np.sign(x)

    def backward(self, grad_output):
        return np.zeros(self.x_shape)


class Sin(Function):
    def forward(self, x):
        self.x = x
        return np.sin(x)

    def backward(self, grad_output):
        return grad_output * np.cos(self.x)


class Cos(Function):
    def forward(self, x):
        self.x = x
        return np.cos(x)

    def backward(self, grad_output):
        return -grad_output * np.sin(self.x)


class Tan(Function):
    def forward(self, x):
        self.x = x
        self.out = np.tan(x)
        return self.out

    def backward(self, grad_output):
        return grad_output / (np.cos(self.x) ** 2)


class Arcsin(Function):
    def forward(self, x):
        self.x = x
        return np.arcsin(x)

    def backward(self, grad_output):
        return grad_output / np.sqrt(1 - self.x**2 + 1e-12)


class Arccos(Function):
    def forward(self, x):
        self.x = x
        return np.arccos(x)

    def backward(self, grad_output):
        return -grad_output / np.sqrt(1 - self.x**2 + 1e-12)


class Arctan(Function):
    def forward(self, x):
        self.x = x
        return np.arctan(x)

    def backward(self, grad_output):
        return grad_output / (1 + self.x**2)


class Sinh(Function):
    def forward(self, x):
        self.x = x
        return np.sinh(x)

    def backward(self, grad_output):
        return grad_output * np.cosh(self.x)


class Cosh(Function):
    def forward(self, x):
        self.x = x
        return np.cosh(x)

    def backward(self, grad_output):
        return grad_output * np.sinh(self.x)


class Log2(Function):
    def forward(self, x):
        self.x = x
        return np.log2(x)

    def backward(self, grad_output):
        return grad_output / (self.x * np.log(2))


class Log10(Function):
    def forward(self, x):
        self.x = x
        return np.log10(x)

    def backward(self, grad_output):
        return grad_output / (self.x * np.log(10))


class Log1p(Function):
    """log(1 + x) — numerically stable for small x"""

    def forward(self, x):
        self.x = x
        return np.log1p(x)

    def backward(self, grad_output):
        return grad_output / (1 + self.x)


class Expm1(Function):
    """exp(x) - 1 — numerically stable for small x"""

    def forward(self, x):
        self.x = x
        return np.expm1(x)

    def backward(self, grad_output):
        return grad_output * np.exp(self.x)


class Ceil(Function):
    """Ceiling — zero gradient (non-differentiable, subgradient 0)"""

    def forward(self, x):
        self.x_shape = x.shape
        return np.ceil(x)

    def backward(self, grad_output):
        return np.zeros(self.x_shape)


class Floor(Function):
    """Floor — zero gradient"""

    def forward(self, x):
        self.x_shape = x.shape
        return np.floor(x)

    def backward(self, grad_output):
        return np.zeros(self.x_shape)


class Round(Function):
    """Round — zero gradient"""

    def forward(self, x):
        self.x_shape = x.shape
        return np.round(x)

    def backward(self, grad_output):
        return np.zeros(self.x_shape)


class Erf(Function):
    """Gauss error function — used in GELU"""

    def forward(self, x):
        from scipy.special import erf

        self.x = x
        return erf(x)

    def backward(self, grad_output):
        return grad_output * (2 / np.sqrt(np.pi)) * np.exp(-(self.x**2))


class Cumsum(Function):
    def __init__(self, *inputs, axis=None):
        super().__init__(*inputs)
        self.axis = axis

    def forward(self, x):
        return np.cumsum(x, axis=self.axis)

    def backward(self, grad_output):
        # reverse cumsum = flip → cumsum → flip
        if self.axis is None:
            g = grad_output.flatten()
            return np.flip(np.cumsum(np.flip(g))).reshape(grad_output.shape)
        return np.flip(
            np.cumsum(np.flip(grad_output, axis=self.axis), axis=self.axis),
            axis=self.axis,
        ).copy()


class Cumprod(Function):
    def __init__(self, *inputs, axis=None):
        super().__init__(*inputs)
        self.axis = axis

    def forward(self, x):
        self.x = x
        self.out = np.cumprod(x, axis=self.axis)
        return self.out

    def backward(self, grad_output):
        # d/dx_i cumprod = sum_j>=i (cumprod_j / x_i) * grad_j
        # stable via: total_prod / x[i] but handle zeros carefully
        x, out = self.x, self.out
        if self.axis is None:
            x = x.flatten()
            out = out.flatten()
            grad_output = grad_output.flatten()

        # Use the log-trick for numerical stability
        grad = np.zeros_like(x)
        n = x.shape[0] if self.axis is None else x.shape[self.axis]
        for i in range(n):
            idx = [slice(None)] * x.ndim
            if self.axis is not None:
                idx[self.axis] = i
            idx = tuple(idx)
            # grad[i] = sum_{j>=i} grad_out[j] * out[j] / x[i]
            for j in range(i, n):
                jdx = [slice(None)] * x.ndim
                if self.axis is not None:
                    jdx[self.axis] = j
                jdx = tuple(jdx)
                denom = x[idx] + 1e-12 * (x[idx] == 0)
                grad[idx] = grad[idx] + grad_output[jdx] * out[jdx] / denom

        return grad.reshape(self.x.shape)


class ArgMax(Function):
    """Non-differentiable — returns integer indices, zero gradient"""

    def __init__(self, *inputs, axis=None):
        super().__init__(*inputs)
        self.axis = axis

    def forward(self, x):
        self.x_shape = x.shape
        return np.argmax(x, axis=self.axis).astype(np.float32)

    def backward(self, grad_output):
        return np.zeros(self.x_shape)


class ArgMin(Function):
    def __init__(self, *inputs, axis=None):
        super().__init__(*inputs)
        self.axis = axis

    def forward(self, x):
        self.x_shape = x.shape
        return np.argmin(x, axis=self.axis).astype(np.float32)

    def backward(self, grad_output):
        return np.zeros(self.x_shape)


class TopK(Function):
    """
    Return top-k values and indices along last axis.
    Returns values only (indices returned separately as non-differentiable).
    """

    def __init__(self, *inputs, k, axis=-1, largest=True):
        super().__init__(*inputs)
        self.k = k
        self.axis = axis
        self.largest = largest

    def forward(self, x):
        self.x_shape = x.shape
        if self.largest:
            idx = np.argpartition(x, -self.k, axis=self.axis)
            idx = np.take(idx, np.arange(-self.k, 0), axis=self.axis)
        else:
            idx = np.argpartition(x, self.k, axis=self.axis)
            idx = np.take(idx, np.arange(self.k), axis=self.axis)

        self.indices = idx
        values = np.take_along_axis(x, idx, axis=self.axis)
        return values

    def backward(self, grad_output):
        grad = np.zeros(self.x_shape)
        np.add.at(
            grad,
            tuple(
                self.indices if i == self.axis else slice(None)
                for i in range(grad.ndim)
            ),
            grad_output,
        )
        return grad
