from ...tensor import functional as F
from .base import Base


class none(Base):
    def forward(self, x):
        return x


class ReLU(Base):
    def forward(self, x):
        return F.relu(x)


class Sigmoid(Base):
    def forward(self, x):
        return F.sigmoid(x)


class Tanh(Base):
    def forward(self, x):
        return x.tanh()


class Softmax(Base):
    def forward(self, x):
        return F.softmax(x)
class LogSoftmax(Base):
    def forward(self, x):
        return F.log_softmax(x)


class LeakyReLU(Base):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.leakyrelu(x, alpha=self.alpha)


class ELU(Base):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.elu(x, alpha=self.alpha)


class Softplus(Base):
    def forward(self, x):
        return F.softplus(x)
