import numpy as np
from .base import Base
from ...tensor import factory as init


class BatchNorm2D(Base):
    """
    Batch Normalization for 2D inputs (N, C, H, W).
    Normalizes across the batch dimension for each channel.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

    def build(self, in_shape):
        # Learnable parameters
        self.gamma = self.add_parameter(
            "gamma", init.tensor(np.ones((self.num_features,)), requires_grad=True)
        )
        self.beta = self.add_parameter(
            "beta", init.tensor(np.zeros((self.num_features,)), requires_grad=True)
        )

        # Running statistics (not trainable)
        self.running_mean = np.zeros((self.num_features,))
        self.running_var = np.ones((self.num_features,))

    def forward(self, x):
        # x shape: (N, C, H, W)
        if self.training:
            # Calculate batch statistics
            # Mean over (N, H, W) dimensions, keep C
            batch_mean = x.data.mean(axis=(0, 2, 3))
            batch_var = x.data.var(axis=(0, 2, 3))

            # Update running statistics
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var

            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics during inference
            mean = self.running_mean
            var = self.running_var

        # Normalize: reshape for broadcasting
        mean_t = init.tensor(mean.reshape(1, -1, 1, 1))
        var_t = init.tensor(var.reshape(1, -1, 1, 1))

        x_normalized = (x - mean_t) / init.tensor(np.sqrt(var_t.data + self.eps))

        # Scale and shift
        gamma = self.gamma.reshape(1, -1, 1, 1)
        beta = self.beta.reshape(1, -1, 1, 1)

        return x_normalized * gamma + beta


