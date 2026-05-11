import numpy as np
from .base import Base
from ...tensor import Tensor
from ...tensor import factory as init


class PositionalEmbedding(Base):
    """
    Learnable positional embeddings for sequence models.
    Adds position information to token embeddings.

    Args:
        max_len: Maximum sequence length
        embedding_dim: Dimension of embeddings (should match token embedding dim)

    Shape:
        - Input: (batch_size, seq_len, embedding_dim)
        - Output: (batch_size, seq_len, embedding_dim)

    Example:
        >>> token_emb = Embedding(vocab_size=1000, embedding_dim=512)
        >>> pos_emb = PositionalEmbedding(max_len=100, embedding_dim=512)
        >>>
        >>> indices = ([[1, 4, 2, 8, 9]], requires_grad=False)
        >>> x = token_emb(indices)  # (1, 5, 512)
        >>> x = pos_emb(x)  # Add positional information
    """

    def __init__(self, max_len, embedding_dim):
        super().__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim

        # Initialize positional embeddings
        pos_emb = np.random.randn(max_len, embedding_dim) * 0.02
        self.pos_embedding = self.add_parameter(
            "pos_embedding", init.tensor(pos_emb, requires_grad=True)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embedding_dim)

        Returns:
            x + positional embeddings
        """
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.max_len}"
            )

        # Get positional embeddings for this sequence length
        pos_emb = self.pos_embedding[:seq_len, :]  # (seq_len, embedding_dim)

        # Broadcast and add to input
        return x + pos_emb.reshape(1, seq_len, self.embedding_dim)


# ============================================================================
# SINUSOIDAL POSITIONAL ENCODING (Fixed, not learnable)
# ============================================================================


class SinusoidalPositionalEncoding(Base):
    """
    Fixed sinusoidal positional encoding (from "Attention is All You Need").
    Uses sine and cosine functions of different frequencies.

    Args:
        max_len: Maximum sequence length
        embedding_dim: Dimension of embeddings

    Shape:
        - Input: (batch_size, seq_len, embedding_dim)
        - Output: (batch_size, seq_len, embedding_dim)
    """

    def __init__(self, max_len, embedding_dim):
        super().__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim

        # Create fixed positional encoding
        pe = np.zeros((max_len, embedding_dim))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(
            np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Store as non-trainable tensor
        self.pe = init.tensor(pe, requires_grad=False)

    def forward(self, x):
        """Add fixed positional encoding to input"""

        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.max_len}"
            )

        # Get positional encoding for this sequence length
        pe = self.pe[:seq_len, :]  # (seq_len, embedding_dim)

        # Add to input
        return x + pe.reshape(1, seq_len, self.embedding_dim)
