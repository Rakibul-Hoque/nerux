import numpy as np
from ..functional.function import Function


# ============================================================================
# EMBEDDING FUNCTION (with autograd support)
# ============================================================================
class EmbeddingFunction(Function):
    def forward(self, indices, weight):
        """
        indices : (*) integer indices
        weight  : (vocab_size, embedding_dim)
        """
        self.indices      = indices.astype(int)
        self.weight_shape = weight.shape
        return weight[self.indices]          # fast fancy indexing

    def backward(self, grad_output):
        """
        grad_output : (*indices.shape, embedding_dim)
        Only weight receives a gradient; indices are discrete.
        """
        grad_weight  = np.zeros(self.weight_shape)
        indices_flat = self.indices.flatten()
        grad_flat    = grad_output.reshape(-1, self.weight_shape[1])

        # np.add.at correctly accumulates repeated indices
        # (direct indexed assignment like grad[idx] += g would only keep the last write)
        np.add.at(grad_weight, indices_flat, grad_flat)

        return None, grad_weight    # None for indices — no gradient needed

 