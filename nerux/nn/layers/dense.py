import numpy as np
from ...tensor import Tensor
from .base import Base
from . import none, ReLU, LeakyReLU, ELU


class Dense(Base):
    def __init__(self, out_features, activation=none()):
        super().__init__()
        self.out_features = out_features
        self.activation = activation

    def build(self, in_shape):
        self.in_features = in_shape[-1]
        rng = np.random.default_rng()
        if isinstance(self.activation, (ReLU, LeakyReLU, ELU)):
            std = np.sqrt(2.0 / self.in_features)
        else:
            std = np.sqrt(2.0 / (self.in_features + self.out_features))
        W = rng.normal(0, std, (self.out_features, self.in_features))
        self.W = self.add_parameter("W", Tensor(W, requires_grad=True))
        B = np.zeros((self.out_features,))
        self.B = self.add_parameter("B", Tensor(B, requires_grad=True))

    def forward(self, x):
        return self.activation((x @ self.W.T) + self.B)
