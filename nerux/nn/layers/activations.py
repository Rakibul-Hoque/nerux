
from .base import Base


class none(Base):
    def forward(self, x):
        return x


class ReLU(Base):
    def forward(self, x):
        return x.relu()


class Sigmoid(Base):
    def forward(self, x):
        return x.sigmoid()


class Tanh(Base):
    def forward(self, x):
        return x.tanh()


class Softmax(Base):
    def forward(self, x):
        return x.softmax()


class LeakyReLU(Base):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x.leakyrelu(alpha=self.alpha)


class ELU(Base):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x.elu(alpha=self.alpha)


class Softplus(Base):
    def forward(self, x):
        return x.softplus()
