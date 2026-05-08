

import numpy as np
from ..function import Function


class ReLU(Function):
    def forward(self, a):
        self.mask = a > 0
        return a * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask


class Sigmoid(Function):
    def forward(self, a):
      out = np.where(a >= 0,
          1 / (1 + np.exp(-a)),
          np.exp(a) / (1 + np.exp(a)))
      self.saved_out = out
      return out

    def backward(self, grad_output):
        return grad_output * self.saved_out * (1 - self.saved_out)


class Tanh(Function):
    def forward(self, a):
        out = np.tanh(a)
        self.saved_out = out
        return out

    def backward(self, grad_output):
        return grad_output * (1 - self.saved_out**2)


class Softmax(Function):
    def __init__(self, *inputs, axis=-1):
        super().__init__(*inputs)
        self.axis = axis

    def forward(self, a):
        exp_a = np.exp(a - np.max(a, axis=self.axis, keepdims=True))
        self.out = exp_a / np.sum(exp_a, axis=self.axis, keepdims=True)
        return self.out

    def backward(self, grad_output):
        y = self.out
        dot = np.sum(grad_output * y, axis=self.axis, keepdims=True)
        return y * (grad_output - dot)


class LeakyReLU(Function):
    def __init__(self, *inputs, alpha=0.01):
        super().__init__(*inputs)
        self.alpha = alpha

    def forward(self, a):
        self.mask = a > 0
        return np.where(self.mask, a, self.alpha * a)

    def backward(self, grad_output):
        grad_input = grad_output * np.where(self.mask, 1, self.alpha)
        return grad_input


class ELU(Function):
    def __init__(self, *inputs, alpha=0.01):
        super().__init__(*inputs)
        self.alpha = alpha

    def forward(self, a):
        self.mask = a > 0
        self.saved_a = a
        return np.where(self.mask, a, self.alpha * (np.exp(a) - 1))

    def backward(self, grad_output):
        grad = np.where(self.mask, 1.0, self.alpha * np.exp(self.saved_a))
        return grad_output * grad


class Softplus(Function):
    def forward(self, a):
        self.saved_a = a
        return np.log(1 + np.exp(a))

    def backward(self, grad_output):
        return grad_output / (1 + np.exp(-self.saved_a))   