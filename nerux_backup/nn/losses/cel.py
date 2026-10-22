from ...tensor import Tensor
import numpy as np


class CrossEntropyLoss:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, pred: Tensor, target: Tensor):
        """
        Supports:
            pred:   (B, V)
                    (B, L, V)
            target: (B)
                    (B, L)
                    (B, V)   one-hot
                    (B, L, V) one-hot
        """
        pred = pred.clip(self.eps, 1 - self.eps)
        log_probs = pred.log()

        # --------------------------------
        # CASE 1: One-hot targets
        # --------------------------------
        if target.ndim == pred.ndim:
            # shapes match: (B,V) or (B,L,V)
            loss = -(target * log_probs).sum(axis=-1).mean()
            return loss

        # --------------------------------
        # CASE 2: Integer class targets
        # pred: (B,V) or (B,L,V)
        # target: (B) or (B,L)
        # --------------------------------

        # Flatten batch/sequence for uniform processing
        if pred.ndim == 3:
            B, L, V = pred.shape
            log_probs = log_probs.reshape(B * L, V)
            target = target.reshape(B * L)
        else:
            B, V = pred.shape
            log_probs = log_probs.reshape(B, V)
            target = target.reshape(B)
        
        N = target.shape[0]
        
        # ✅ Gather correct class log_probs
        indices = np.arange(N)
        correct_log_probs = log_probs[indices, target.data]
        
        loss = -correct_log_probs.mean()
        return loss






class CrossEntropyLoss3:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, pred: Tensor, target: Tensor):
        pred = pred.clip(self.eps, 1 - self.eps)
        if target.ndim == 1:
            N = target.shape[0]
            log_probs = pred.log()
            loss = -log_probs[np.arange(N), target].mean()
        else:
            loss = -(target * pred.log()).sum(axis=1).mean()
        return loss
