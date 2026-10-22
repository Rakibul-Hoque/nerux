from ...tensor import Tensor


class CCELoss:
    def __call__(self, pred: Tensor, target: Tensor, eps=1e-8):
        pred = pred.clip(eps, 1 - eps)  # prevent log(0)
        loss = -(target * pred.log()).mean()
        return loss
