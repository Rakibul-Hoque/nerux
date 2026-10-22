import numpy as np
from ...tensor import Tensor
from ...tensor import functional as F
from ...tensor import factory as init

from .base import Base
from .linear import Linear
from .dropout import Dropout


class MultiHeadAttention(Base):
    def __init__(self, d_model, num_heads, dropout=0.1, bias=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout
        self.use_bias = bias

        self.q_layer = None
        self.k_layer = None
        self.v_layer = None
        self.W_o_layer = None
        self.dropout = None

    def build(self, in_shape):
        input_d_model = in_shape[-1]
        if input_d_model != self.d_model:
            raise ValueError(...)

        # One big projection per role — all d_model -> d_model
        self.q_layer = self.add_layer(
            "q_layer", Linear(self.d_model, bias=self.use_bias)
        )
        self.k_layer = self.add_layer(
            "k_layer", Linear(self.d_model, bias=self.use_bias)
        )
        self.v_layer = self.add_layer(
            "v_layer", Linear(self.d_model, bias=self.use_bias)
        )
        self.W_o_layer = self.add_layer(
            "W_o_layer", Linear(self.d_model, bias=self.use_bias)
        )
        self.dropout = self.add_layer("dropout", Dropout(self.dropout_rate))

    def forward(self, Q, K=None, V=None, mask=None):
        """
        Q, K, V : (batch, seq_len, d_model)
        mask    : (batch, seq_len_k) or
                  (batch, 1, seq_len_k) or
                  (batch, seq_len_q, seq_len_k) or
                  (batch, heads, seq_len_q, seq_len_k
                   — 1 = keep, 0 = mask
        Returns : (batch, seq_len_q, d_model)
        """

        B, S, D = Q.shape

        K = Q if K is None else K
        V = Q if V is None else V

        # 1) Project to d_model (still 3D)
        Q = self.q_layer(Q)  # (B, S, D)
        K = self.k_layer(K)  # (B, S, D)
        V = self.v_layer(V)  # (B, S, D)

        H, d_k = self.num_heads, self.d_k

        # 2) Split into heads: (B, S, D) -> (B, S, H, d_k) -> (B, H, S, d_k)
        Q = Q.reshape(B, S, H, d_k).transpose(axes=(0, 2, 1, 3))
        K = K.reshape(B, S, H, d_k).transpose(axes=(0, 2, 1, 3))
        V = V.reshape(B, S, H, d_k).transpose(axes=(0, 2, 1, 3))

        norm_mask = self._normalise_mask(mask)  # (B, 1, S_q, S_k) or None

        # 3) Attention across all heads at once
        H_out = self._attention(Q, K, V, norm_mask)  # (B, H, S_q, d_k)

        # 4) Merge heads: (B, H, S, d_k) -> (B, S, H, d_k) -> (B, S, D)
        H_out = H_out.transpose(axes=(0, 2, 1, 3)).reshape(B, S, D)

        # 5) Output mixing + dropout
        output = self.W_o_layer(H_out)
        output = self.dropout(output)
        return output

    def _normalise_mask(self, mask):
        """
        Returns a mask shaped to broadcast against scores (B, H, S_q, S_k):
          - None                            → None
          - (B, S_k)                        → (B, 1, 1, S_k)
          - (B, 1, S_k)                     → (B, 1, 1, S_k)
          - (B, S_q, S_k)                   → (B, 1, S_q, S_k)
          - (B, H, S_q, S_k)                → (B, H, S_q, S_k)   (as-is)
        """
        if mask is None:
            return None

        m = mask.data if isinstance(mask, Tensor) else np.array(mask, dtype=float)

        if m.ndim == 2:  # (B, S_k)
            m = m[:, np.newaxis, np.newaxis, :]  # (B, 1, 1, S_k)
        elif m.ndim == 3:  # (B, S_q, S_k)  or (B, 1, S_k)
            m = m[:, np.newaxis, :, :]  # (B, 1, S_q|1, S_k)
        # ndim == 4: leave as (B, H, S_q, S_k)
        return m.astype(float)

    def _attention(self, Q, K, V, mask=None):
        """
        Q, K, V : (B, H, S, d_k)
        mask    : broadcastable to (B, H, S_q, S_k), 1=keep 0=mask  or None
        Returns : (B, H, S_q, d_k)
        """


        # (B, H, S, d_k) @ (B, H, d_k, S) -> (B, H, S, S)
        K_T = K.transpose(axes=(0, 1, 3, 2))
        scores = (Q @ K_T) * (1.0 / np.sqrt(self.d_k))

        if mask is not None:
            mask_t = init.tensor(mask, requires_grad=False)
            penalty = init.tensor((1.0 - mask) * -1e9, requires_grad=False)
            scores = scores * mask_t + penalty

        attn = scores.softmax()  # softmax over last axis (S_k)
        return attn @ V  # (B, H, S_q, d_k)
