import numpy as np
from .base import Base
from ..parameter import Parameter
from ..functional.batchnorm import BatchNormFunction


class BatchNorm1D(Base):
    """
    Batch Normalization for (N, C) or (N, C, L) inputs.
    Normalizes over the batch (and spatial) dims for each channel.
    Gradients flow through the batch statistics (standard BatchNorm).
    """

    _VALID_NDIM = (2, 3)

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

    def build(self, in_shape):
        self.gamma = self.add_parameter(
            "gamma", Parameter(np.ones((self.num_features,)))
        )
        self.beta = self.add_parameter(
            "beta", Parameter(np.zeros((self.num_features,)))
        )

        # Running statistics (not trainable); updated in place by the function
        self.running_mean = np.zeros((self.num_features,))
        self.running_var = np.ones((self.num_features,))

    def forward(self, x):
        if len(x.shape) not in self._VALID_NDIM:
            raise ValueError(
                f"BatchNorm1D expects (N, C) or (N, C, L) inputs, got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"BatchNorm1D expects {self.num_features} channels, got {x.shape[1]}"
            )
        return BatchNormFunction.apply(
            x,
            self.gamma,
            self.beta,
            running_mean=self.running_mean,
            running_var=self.running_var,
            eps=self.eps,
            momentum=self.momentum,
            training=self.training,
        )
