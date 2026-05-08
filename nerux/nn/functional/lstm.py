import numpy as np
from .function import Function


# ── Fused single-step LSTM Function ──────────────────────────────────────────
# One Function per timestep keeps the autograd graph flat and correct.
# All gate math happens inside forward/backward in raw numpy — no Tensor ops
# inside here, so there is zero risk of touching or breaking the Tensor autograd.


class LSTMCellFunction(Function):
    """
    Inputs:
        x   : (N, input_size)
        h   : (N, proj_size)  if proj_size > 0, else (N, hidden_size)
        c   : (N, hidden_size)  — always hidden_size, never projected
        W_ih: (4*hidden_size, input_size)
        W_hh: (4*hidden_size, proj_size)  if proj_size > 0, else (4*H, H)
        b   : (4*hidden_size,)

    Output packed (3, N, hidden_size):
        [0] = h_new  (N, hidden_size) — before projection
        [1] = c_new  (N, hidden_size)
        [2] = c_old  (N, hidden_size)
    """

    def forward(self, x, h, c, W_ih, W_hh, b):
        # H must come from W_ih, not h.shape
        # W_ih shape is (4*hidden_size, input_size)
        # h shape is (N, proj_size) when projection is active — NOT hidden_size
        H = W_ih.shape[0] // 4  # true hidden_size always
        N = x.shape[0]

        gates = x @ W_ih.T + h @ W_hh.T + b
        gates = np.clip(gates, -15.0, 15.0)

        i = _sigmoid(gates[:, 0 * H : 1 * H])
        f = _sigmoid(gates[:, 1 * H : 2 * H])
        g = np.tanh(gates[:, 2 * H : 3 * H])
        o = _sigmoid(gates[:, 3 * H : 4 * H])

        c_new = f * c + i * g  # c is (N, H), f/i/g are (N, H) — matches
        c_new = np.clip(c_new, -15.0, 15.0)
        h_new = o * np.tanh(c_new)  # h_new is (N, H) — projection happens in layer

        self._cache = (x, h, c, W_ih, W_hh, i, f, g, o, c_new, h_new, N, H)
        return np.stack([h_new, c_new, c], axis=0)

    def backward(self, grad_packed):
        x, h, c, W_ih, W_hh, i, f, g, o, c_new, h_new, N, H = self._cache

        dh_new = grad_packed[0]
        dc_new = grad_packed[1]

        tanh_c_new = np.tanh(c_new)  # safe: c_new was clipped in forward

        do = dh_new * tanh_c_new
        dc_new = dc_new + dh_new * o * (1.0 - tanh_c_new**2)

        df = dc_new * c
        di = dc_new * g
        dg = dc_new * i
        dc_prev = dc_new * f

        di_pre = di * _sigmoid_grad(i)
        df_pre = df * _sigmoid_grad(f)
        dg_pre = dg * (1.0 - g**2)
        do_pre = do * _sigmoid_grad(o)

        d_gates = np.concatenate([di_pre, df_pre, dg_pre, do_pre], axis=1)

        dW_ih = d_gates.T @ x
        dW_hh = d_gates.T @ h
        db = d_gates.sum(axis=0)
        dx = d_gates @ W_ih
        dh = d_gates @ W_hh

        return dx, dh, dc_prev, dW_ih, dW_hh, db


def _sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _sigmoid_grad(s):
    return s * (1.0 - s)


def _safe_tanh(x):
    return np.tanh(np.clip(x, -15.0, 15.0))
