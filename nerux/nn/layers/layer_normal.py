import numpy as np
from ...tensor import Tensor
from .base import Base
from ...tensor import factory as init


class LayerNorm(Base):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

    def build(self, in_shape):
        if tuple(in_shape[-len(self.normalized_shape) :]) != self.normalized_shape:
            raise ValueError(
                f"LayerNorm normalized_shape={self.normalized_shape} does not match input tail shape {in_shape}"
            )

        gamma = np.ones(self.normalized_shape, dtype=np.float32)
        beta = np.zeros(self.normalized_shape, dtype=np.float32)

        self.gamma = self.add_parameter("gamma", init.tensor(gamma, requires_grad=True))
        self.beta = self.add_parameter("beta", init.tensor(beta, requires_grad=True))

    def forward(self, x: Tensor):
        reduce_ndims = len(self.normalized_shape)
        axes = tuple(range(len(x.shape) - reduce_ndims, len(x.shape)))

        mean = x.mean(axis=axes, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)

        x_hat = (x - mean) / (var + self.eps).sqrt()

        return self.gamma * x_hat + self.beta
