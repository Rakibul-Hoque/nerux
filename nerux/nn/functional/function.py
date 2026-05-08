import numpy as np

from ...tensor.function import Function as FN





class Function(FN):
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
