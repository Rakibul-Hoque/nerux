# tensor/nn/losses/__init__.py
from .mse import MSELoss
from .mae import MAELoss
from .bce import BCELoss
from .cce import CCELoss
from .cel import CrossEntropyLoss

__all__ = ["MSELoss", "BCELoss", "CCELoss"]
