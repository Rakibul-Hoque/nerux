from ..functional.embedding import EmbeddingFunction
import numpy as np

from .base import Base
from ...tensor import factory as init
from ...tensor import Tensor





class Embedding(Base):
    """
    Embedding layer that maps integer indices to dense vectors.

    This is typically used as the first layer in NLP models to convert
    token IDs into dense vector representations.

    Args:
        vocab_size: Size of the vocabulary (number of unique tokens)
        embedding_dim: Dimension of the embedding vectors
        padding_idx: If specified, the embedding at this index will be zero and won't be updated
        max_norm: If specified, embeddings are normalized to have at most this L2 norm
        scale_grad_by_freq: Scale gradients by frequency of words in minibatch

    Shape:
        - Input: (batch_size, seq_len) or any shape with integer indices
        - Output: (*input.shape, embedding_dim)

    Examples:
        >>> # Vocabulary of 1000 words, embedding dimension of 300
        >>> embedding = Embedding(vocab_size=1000, embedding_dim=300)
        >>>
        >>> # Input: batch of 32 sequences, each with 10 tokens
        >>> indices = Tensor([[1, 4, 2, 8, 9, 3, 5, 7, 6, 0],
        >>>                   [5, 2, 1, 9, 8, 4, 3, 7, 6, 0], ...], requires_grad=False)
        >>>
        >>> # Output: (32, 10, 300)
        >>> embedded = embedding(indices)
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm

        # Standard normal init — same as PyTorch default
        # Xavier is wrong here: embeddings are a lookup table, not a linear layer
        weight = np.random.normal(0.0, 1.0, (vocab_size, embedding_dim))

        if padding_idx is not None:
            weight[padding_idx] = 0.0

        self.weight = self.add_parameter(
            "weight", init.tensor(weight, requires_grad=True)
        )

    def forward(self, indices):
        indices_np = indices.data if isinstance(indices, Tensor) else indices
        indices_np = indices_np.astype(np.int32)  # ensure int before any check

        # Clamp out-of-range to UNK rather than crashing
        # (handles the dummy build tensor and any edge cases)
        indices_np = np.clip(indices_np, 0, self.vocab_size - 1)

        # Wrap back into a non-grad Tensor for Function.apply
        indices_tensor = init.tensor(indices_np.astype(np.float32), requires_grad=False)

        embedded = EmbeddingFunction.apply(indices_tensor, self.weight)

        if self.max_norm is not None:
            embedded = self._apply_max_norm(embedded)

        if self.padding_idx is not None:
            mask = (indices_np != self.padding_idx).astype(np.float32)
            mask = init.tensor(
                np.expand_dims(
                    mask, -1
                ),  # (batch, seq, 1) — broadcasts over embedding_dim
                requires_grad=False,
            )
            embedded = embedded * mask

        return embedded

    def _apply_max_norm(self, embedded):
        norms = (embedded * embedded).sum(axis=-1, keepdims=True).sqrt()
        scale = (self.max_norm / (norms + 1e-8)).clip(0, 1)
        return embedded * scale

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None):
        """
        Create embedding layer from pretrained embeddings.

        Args:
            embeddings: numpy array or Tensor of shape (vocab_size, embedding_dim)
            freeze: If True, embeddings won't be updated during training
            padding_idx: Optional padding index

        Returns:
            Embedding layer with pretrained weights
        """
        if isinstance(embeddings, Tensor):
            embeddings = embeddings.data
        vocab_size, embedding_dim = embeddings.shape
        layer = cls(vocab_size, embedding_dim, padding_idx=padding_idx)
        layer.weight = init.tensor(embeddings.copy(), requires_grad=not freeze)
        layer._parameters["weight"] = layer.weight
        return layer
