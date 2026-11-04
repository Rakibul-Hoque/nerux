from ..tensor import Tensor


class Parameter(Tensor):
    __slots__ = ("_module",)

    _is_param = True

    def __init__(self, data, dtype=None, requires_grad=True):
        super().__init__(
            data,
            dtype=dtype,
            requires_grad=requires_grad,
            creator=None,
        )

        self._module = None

    @classmethod
    def from_tensor(cls, t, requires_grad=True):
        return cls(t.data.copy(), dtype=None, requires_grad=requires_grad)

    def __repr__(self):
        return (
            f"Parameter(shape={tuple(self.shape)}, "
            f"dtype={self.dtype}, "
            f"requires_grad={self.requires_grad})"
        )


def to_parameter(t: Tensor, requires_grad) -> Parameter:
    return Parameter.from_tensor(t, requires_grad)
