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

        self.head_q_layers = []
        self.head_k_layers = []
        self.head_v_layers = []
        self.W_o = None
        self.dropout = None

    def build(self, in_shape):
        input_d_model = in_shape[-1]
        if input_d_model != self.d_model:
            raise ValueError(
                f"Input dimension {input_d_model} doesn't match d_model {self.d_model}"
            )

        for h in range(self.num_heads):
            self.head_q_layers.append(
                self.add_layer(f"head_q_{h}", Linear(self.d_k, bias=self.use_bias))
            )
            self.head_k_layers.append(
                self.add_layer(f"head_k_{h}", Linear(self.d_k, bias=self.use_bias))
            )
            self.head_v_layers.append(
                self.add_layer(f"head_v_{h}", Linear(self.d_k, bias=self.use_bias))
            )

        self.W_o = self.add_layer("W_o", Linear(self.d_model, bias=self.use_bias))
        self.dropout = self.add_layer("dropout", Dropout(self.dropout_rate))

    def forward(self, Q, K=None, V=None, mask=None):
        """
        Q, K, V : (batch, seq_len, d_model)
        mask    : (batch, seq_len_k) or (batch, 1, seq_len_k)
                  or (batch, seq_len_q, seq_len_k)   — 1 = keep, 0 = mask
        Returns : (batch, seq_len_q, d_model)
        """
        K = Q if K is None else K
        V = Q if V is None else V

        # Normalise mask to (batch, seq_len_q, seq_len_k) once,
        # then reuse across all heads — no per-head slicing needed
        norm_mask = self._normalise_mask(mask, Q)

        head_outputs = []
        for h in range(self.num_heads):
            Q_h = self.head_q_layers[h](Q)
            K_h = self.head_k_layers[h](K)
            V_h = self.head_v_layers[h](V)
            head_outputs.append(self._attention(Q_h, K_h, V_h, norm_mask))

        concat = F.concat(head_outputs, axis=2)
        output = self.W_o(concat)
        output = self.dropout(output)
        return output

    # ── helpers ──────────────────────────────────────────────────────────────

    def _normalise_mask(self, mask, Q):
        """
        Accept masks in any of these shapes and return (batch, seq_q, seq_k):
          - None
          - (batch, seq_k)                  → broadcast over seq_q
          - (batch, 1, seq_k)               → broadcast over seq_q
          - (batch, seq_q, seq_k)           → use as-is
          - (batch, num_heads, seq_q, seq_k)→ average over heads (all heads
                                              should be identical in practice)
        """
        if mask is None:
            return None

        m = mask.data if isinstance(mask, Tensor) else np.array(mask, dtype=float)

        if m.ndim == 2:
            # (batch, seq_k) → (batch, 1, seq_k)
            m = m[:, np.newaxis, :]
        elif m.ndim == 4:
            # (batch, heads, seq_q, seq_k) → take first head
            # (all heads carry the same padding mask)
            m = m[:, 0, :, :]
        # now m is (batch, 1|seq_q, seq_k) — fine for broadcasting in _attention
        return m.astype(float)

    def _attention(self, Q, K, V, mask=None):
        """
        Scaled dot-product attention.
        Q, K, V : (batch, seq, d_k)
        mask    : (batch, 1|seq_q, seq_k)  float  1=keep 0=mask  or None
        """
        d_k = Q.shape[-1]
        K_T = K.transpose(axes=(0, 2, 1))
        scores = (Q @ K_T) * (1.0 / np.sqrt(d_k))  # avoid Tensor / scalar division

        if mask is not None:
            # Add large negative where mask == 0 so softmax → ~0
            penalty = init.tensor((1.0 - mask) * -1e9, requires_grad=False)
            mask_t = init.tensor(mask, requires_grad=False)
            scores = scores * mask_t + penalty

        attn_weights = scores.softmax()
        return attn_weights @ V
