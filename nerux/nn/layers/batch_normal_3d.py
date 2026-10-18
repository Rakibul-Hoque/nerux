import numpy as np
from .base import Base
from ...tensor import factory as init


class BatchNorm3D(Base):
    """
    Batch Normalization for 5D inputs (N, C, D, H, W).
    Normalizes across (N, D, H, W) for each channel.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

    def build(self, in_shape):
        self.gamma = self.add_parameter(
            "gamma", init.tensor(np.ones((self.num_features,)), requires_grad=True)
        )
        self.beta = self.add_parameter(
            "beta", init.tensor(np.zeros((self.num_features,)), requires_grad=True)
        )

        self.running_mean = np.zeros((self.num_features,))
        self.running_var = np.ones((self.num_features,))

    def forward(self, x):
        # x shape: (N, C, D, H, W)
        if x.data.ndim != 5:
            raise ValueError(f"BatchNorm3D expects (N, C, D, H, W), got {x.data.shape}")

        axes = (0, 2, 3, 4)
        bshape = (1, -1, 1, 1, 1)

        if self.training:
            batch_mean = x.data.mean(axis=axes)
            batch_var = x.data.var(axis=axes)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var

            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        mean_t = init.tensor(mean.reshape(bshape))
        var_t = init.tensor(var.reshape(bshape))

        x_normalized = (x - mean_t) / init.tensor(np.sqrt(var_t.data + self.eps))

        gamma = self.gamma.reshape(*bshape)
        beta = self.beta.reshape(*bshape)

        return x_normalized * gamma + beta
