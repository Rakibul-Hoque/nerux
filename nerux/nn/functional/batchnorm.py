import numpy as np
from .function import Function


class BatchNormFunction(Function):
    """
    Batch norm for any (N, C, *spatial) / (N, C) input, with a full analytic
    backward INCLUDING the dependence of the batch mean/var on x.

    running_mean / running_var are float numpy arrays updated IN PLACE
    during training.
    """

    def forward(
        self,
        x,
        gamma,
        beta,
        running_mean=None,
        running_var=None,
        eps=1e-5,
        momentum=0.1,
        training=True,
    ):
        if x.ndim < 2:
            raise ValueError(f"BatchNorm expects (N, C, ...), got shape {x.shape}")
        axes = (0,) + tuple(range(2, x.ndim))
        shape = (1, -1) + (1,) * (x.ndim - 2)

        if training:
            if x.size // x.shape[1] < 2:
                raise ValueError(
                    "BatchNorm needs more than 1 value per channel in training mode"
                )
            mean = x.mean(axis=axes)
            var = x.var(axis=axes)
            if running_mean is not None:
                running_mean *= 1 - momentum
                running_mean += momentum * mean
                running_var *= 1 - momentum
                running_var += momentum * var
        else:
            mean, var = running_mean, running_var

        invstd = 1.0 / np.sqrt(var + eps)
        xhat = (x - mean.reshape(shape)) * invstd.reshape(shape)
        out = xhat * gamma.reshape(shape) + beta.reshape(shape)

        self.xhat, self.gamma, self.invstd = xhat, gamma, invstd
        self.axes, self.shape, self.training = axes, shape, training
        self.m = x.size // x.shape[1]
        return out

    def backward(self, grad_output):
        g = grad_output
        axes, shape, m = self.axes, self.shape, self.m

        grad_beta = g.sum(axis=axes)
        grad_gamma = (g * self.xhat).sum(axis=axes)
        scale = (self.gamma * self.invstd).reshape(shape)

        if self.training:
            grad_x = (scale / m) * (
                m * g
                - grad_beta.reshape(shape)
                - self.xhat * grad_gamma.reshape(shape)
            )
        else:
            grad_x = g * scale
        return grad_x, grad_gamma, grad_beta
