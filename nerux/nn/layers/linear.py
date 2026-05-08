import numpy as np
from ...tensor import Tensor
from .base import Base


class Linear(Base):
    def __init__(self, out_features, bias=True):
        super().__init__()
        self.out_features = out_features
        self.is_bias = bias

    def build(self, in_shape):
        self.in_features = in_shape[-1]
        limit = np.sqrt(6 / (self.in_features + self.out_features))
        W = np.random.uniform(-limit, limit, (self.out_features, self.in_features))
        self.W = self.add_parameter("W", Tensor(W, requires_grad=True))

        if self.is_bias:
            B = np.zeros((self.out_features,))
            self.B = self.add_parameter("B", Tensor(B, requires_grad=True))
        else:
            self.B = None

    def forward(self, x):
        out = x @ self.W.T
        if self.B is not None:
            out = out + self.B
        return out
