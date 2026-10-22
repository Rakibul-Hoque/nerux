from .grad_clip import _clip_grad_norm, _clip_grad_value




class Optimizer:
    def __init__(self, params, lr=0.01, grad_clip=None, grad_clip_mode="norm"):

        self.params = list(params)
        self.lr = lr
        self.grad_clip = grad_clip
        self.grad_clip_mode = grad_clip_mode

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def clip_gradients(self):
        if self.grad_clip is None:
            return None

        if self.grad_clip_mode == "norm":
            return _clip_grad_norm(self.params, self.grad_clip)
        elif self.grad_clip_mode == "value":
            return _clip_grad_value(self.params, self.grad_clip)
        else:
            raise ValueError(f"Unknown grad_clip_mode: {self.grad_clip_mode}")

    def step(self):
        if self.grad_clip is not None:
            self.clip_gradients()
        self._step()

    def _step(self):
        raise NotImplementedError
