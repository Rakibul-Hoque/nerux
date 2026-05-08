# ============================================================================
# MULTI-HEAD ATTENTION - Complete Implementation
# ============================================================================

import numpy as np
from ...tensor import Tensor
from .base import Base


# ============================================================================
# SCALED DOT-PRODUCT ATTENTION (Core Attention Mechanism)
# ============================================================================


class ScaledDotProductAttention(Base):
    """
    Scaled Dot-Product Attention from "Attention is All You Need"

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        dropout: Dropout probability for attention weights

    Shape:
        - Q: (batch, num_heads, seq_len_q, d_k)
        - K: (batch, num_heads, seq_len_k, d_k)
        - V: (batch, num_heads, seq_len_v, d_v)
        - mask: (batch, 1, seq_len_q, seq_len_k) or (batch, num_heads, seq_len_q, seq_len_k)
        - Output: (batch, num_heads, seq_len_q, d_v)
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.attention_weights = None  # Store for visualization

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query tensor (batch, num_heads, seq_len_q, d_k)
            K: Key tensor (batch, num_heads, seq_len_k, d_k)
            V: Value tensor (batch, num_heads, seq_len_v, d_v)
            mask: Optional mask tensor (1 = keep, 0 = mask out)

        Returns:
            output: (batch, num_heads, seq_len_q, d_v)
            attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
        """
        d_k = Q.shape[-1]

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # (batch, num_heads, seq_len_q, d_k) @ (batch, num_heads, d_k, seq_len_k)
        # = (batch, num_heads, seq_len_q, seq_len_k)
        scores = (Q @ K.transpose(axes=(0, 1, 3, 2))) / np.sqrt(d_k)

        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            mask_data = mask.data if isinstance(mask, Tensor) else mask
            # Where mask is 0, set scores to -1e9
            scores = Tensor(np.where(mask_data == 0, -1e9, scores.data))

        # Apply softmax to get attention weights
        attention_weights = scores.softmax()  # Along last dimension (seq_len_k)

        # Store for visualization/analysis
        self.attention_weights = attention_weights

        # Apply dropout if training
        if self.training and self.dropout > 0:
            # Create dropout mask
            dropout_mask = (
                np.random.rand(*attention_weights.shape) > self.dropout
            ).astype(float)
            attention_weights = attention_weights * Tensor(
                dropout_mask / (1 - self.dropout)
            )

        # Apply attention to values
        # (batch, num_heads, seq_len_q, seq_len_k) @ (batch, num_heads, seq_len_k, d_v)
        # = (batch, num_heads, seq_len_q, d_v)
        output = attention_weights @ V

        return output, attention_weights


# ============================================================================
# MULTI-HEAD ATTENTION
# ============================================================================


class MultiHeadAttention(Base):
    """
    Multi-Head Attention mechanism from "Attention is All You Need"

    Projects Q, K, V into multiple heads, applies attention in parallel,
    then concatenates and projects back.

    Args:
        d_model: Dimension of model (embedding dimension)
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in linear projections

    Shape:
        - Q: (batch, seq_len_q, d_model)
        - K: (batch, seq_len_k, d_model)
        - V: (batch, seq_len_v, d_model)
        - Output: (batch, seq_len_q, d_model)

    Example:
        >>> mha = MultiHeadAttention(d_model=512, num_heads=8)
        >>> Q = Tensor.randn((32, 10, 512))  # batch=32, seq_len=10, d_model=512
        >>> K = Tensor.randn((32, 10, 512))
        >>> V = Tensor.randn((32, 10, 512))
        >>> output, attn_weights = mha(Q, K, V)
        >>> # output shape: (32, 10, 512)
    """

    def __init__(self, d_model, num_heads, dropout=0.1, bias=True):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.d_v = d_model // num_heads

        # Linear projections for Q, K, V
        from .dense import Dense

        self.W_q = self.add_layer("W_q", Dense(d_model, bias=bias))
        self.W_k = self.add_layer("W_k", Dense(d_model, bias=bias))
        self.W_v = self.add_layer("W_v", Dense(d_model, bias=bias))

        # Output projection
        self.W_o = self.add_layer("W_o", Dense(d_model, bias=bias))

        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout)

        self.dropout = dropout

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k)

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.shape

        # Reshape to (batch, seq_len, num_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose to (batch, num_heads, seq_len, d_k)
        x = x.transpose(axes=(0, 2, 1, 3))

        return x

    def combine_heads(self, x):
        """
        Inverse of split_heads

        Args:
            x: (batch, num_heads, seq_len, d_k)

        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.shape

        # Transpose to (batch, seq_len, num_heads, d_k)
        x = x.transpose(axes=(0, 2, 1, 3))

        # Reshape to (batch, seq_len, d_model)
        x = x.reshape(batch_size, seq_len, self.d_model)

        return x

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query tensor (batch, seq_len_q, d_model)
            K: Key tensor (batch, seq_len_k, d_model)
            V: Value tensor (batch, seq_len_v, d_model)
            mask: Optional mask tensor (batch, 1, seq_len_q, seq_len_k)

        Returns:
            output: (batch, seq_len_q, d_model)
            attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
        """
        
        # Linear projections
        Q = self.W_q(Q)  # (batch, seq_len_q, d_model)
        K = self.W_k(K)  # (batch, seq_len_k, d_model)
        V = self.W_v(V)  # (batch, seq_len_v, d_model)

        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch, num_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch, num_heads, seq_len_v, d_v)

        # Apply scaled dot-product attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        # attn_output: (batch, num_heads, seq_len_q, d_v)

        # Combine heads
        attn_output = self.combine_heads(attn_output)  # (batch, seq_len_q, d_model)

        # Final linear projection
        output = self.W_o(attn_output)  # (batch, seq_len_q, d_model)

        return output, attention_weights


# ============================================================================
# HELPER: CREATE ATTENTION MASKS
# ============================================================================


def create_padding_mask(seq, pad_idx=0):
    """
    Create mask for padding tokens.

    Args:
        seq: (batch, seq_len) - sequence with padding
        pad_idx: Index of padding token

    Returns:
        mask: (batch, 1, 1, seq_len) - 1 for real tokens, 0 for padding
    """
    seq_data = seq.data if isinstance(seq, Tensor) else seq

    # Create mask: 1 where not padding, 0 where padding
    mask = (seq_data != pad_idx).astype(float)

    # Add dimensions for broadcasting: (batch, 1, 1, seq_len)
    mask = mask[:, np.newaxis, np.newaxis, :]

    return Tensor(mask)


def create_look_ahead_mask(size):
    """
    Create causal mask for autoregressive models (like GPT).
    Prevents attention to future positions.

    Args:
        size: Sequence length

    Returns:
        mask: (1, 1, size, size) - Upper triangular matrix of 1s
    """
    # Lower triangular matrix (can see current and past)
    mask = np.tril(np.ones((size, size)))

    # Add batch and head dimensions
    mask = mask[np.newaxis, np.newaxis, :, :]

    return Tensor(mask)


def create_combined_mask(seq, pad_idx=0):
    """
    Create mask combining padding and look-ahead masks (for decoder).

    Args:
        seq: (batch, seq_len)
        pad_idx: Padding token index

    Returns:
        mask: (batch, 1, seq_len, seq_len)
    """
    seq_len = seq.shape[1]

    # Padding mask
    padding_mask = create_padding_mask(seq, pad_idx)

    # Look-ahead mask
    look_ahead_mask = create_look_ahead_mask(seq_len)

    # Combine: both masks must be 1 for position to be attended to
    combined_mask = Tensor(np.minimum(padding_mask.data, look_ahead_mask.data))

    return combined_mask


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# Example 1: Self-Attention (Q = K = V)
mha = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)

x = Tensor.randn((32, 10, 512))  # (batch=32, seq_len=10, d_model=512)
output, attn_weights = mha(x, x, x)  # Self-attention

print(output.shape)  # (32, 10, 512)
print(attn_weights.shape)  # (32, 8, 10, 10) - 8 heads


# Example 2: Cross-Attention (encoder-decoder attention)
encoder_output = Tensor.randn((32, 20, 512))  # Encoder output
decoder_input = Tensor.randn((32, 10, 512))   # Decoder input

mha = MultiHeadAttention(d_model=512, num_heads=8)
output, attn_weights = mha(
    Q=decoder_input,      # Query from decoder
    K=encoder_output,     # Keys from encoder
    V=encoder_output      # Values from encoder
)

print(output.shape)  # (32, 10, 512) - same as query


# Example 3: With Padding Mask
mha = MultiHeadAttention(d_model=512, num_heads=8)

# Sequence with padding (0 = padding token)
seq = Tensor(np.array([[1, 4, 2, 8, 0, 0],   # 4 real tokens
                       [5, 2, 1, 9, 8, 4]]))  # 6 real tokens

x = Tensor.randn((2, 6, 512))

# Create padding mask
mask = create_padding_mask(seq, pad_idx=0)

output, attn_weights = mha(x, x, x, mask=mask)
# Padding positions won't affect attention


# Example 4: Causal/Look-ahead Mask (for GPT-style models)
mha = MultiHeadAttention(d_model=512, num_heads=8)

x = Tensor.randn((32, 10, 512))

# Create look-ahead mask (can't attend to future)
mask = create_look_ahead_mask(size=10)

output, attn_weights = mha(x, x, x, mask=mask)
# Each position can only attend to itself and previous positions


# Example 5: Complete Transformer Encoder Layer
class TransformerEncoderLayer(Base):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        from .dense import Dense
        from .dropout import Dropout
        from .layer_norm import LayerNorm  # You'll need this
        
        # Multi-head attention
        self.mha = self.add_layer("mha", 
            MultiHeadAttention(d_model, num_heads, dropout))
        
        # Feed-forward network
        self.ffn = self.add_layer("ffn", Dense(d_ff, activation=layers.ReLU()))
        self.ffn_out = self.add_layer("ffn_out", Dense(d_model))
        
        # Layer normalization and dropout
        self.norm1 = self.add_layer("norm1", LayerNorm(d_model))
        self.norm2 = self.add_layer("norm2", LayerNorm(d_model))
        self.dropout1 = self.add_layer("dropout1", Dropout(dropout))
        self.dropout2 = self.add_layer("dropout2", Dropout(dropout))
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)  # Add & Norm
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        ffn_output = self.ffn_out(ffn_output)
        ffn_output = self.dropout2(ffn_output)
        x = self.norm2(x + ffn_output)  # Add & Norm
        
        return x


# Example 6: Complete Transformer Model
class SimpleTransformer(Base):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, 
                 max_len, num_classes, dropout=0.1):
        super().__init__()
        from .embedding import Embedding, PositionalEmbedding
        from .dense import Dense
        from .dropout import Dropout
        
        # Embeddings
        self.token_emb = self.add_layer("token_emb", 
            Embedding(vocab_size, d_model, padding_idx=0))
        self.pos_emb = self.add_layer("pos_emb", 
            PositionalEmbedding(max_len, d_model))
        
        # Encoder layers
        self.encoder_layers = []
        for i in range(num_layers):
            layer = self.add_layer(f"encoder_{i}", 
                TransformerEncoderLayer(d_model, num_heads, d_model*4, dropout))
            self.encoder_layers.append(layer)
        
        # Classification head
        self.dropout = self.add_layer("dropout", Dropout(dropout))
        self.fc = self.add_layer("fc", Dense(num_classes))
    
    def forward(self, x, mask=None):
        # Embeddings
        x = self.token_emb(x)
        x = self.pos_emb(x)
        
        # Encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        # Global average pooling
        x = x.mean(axis=1)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# Usage
model = SimpleTransformer(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    max_len=100,
    num_classes=10,
    dropout=0.1
)

# Input sequences
sequences = Tensor(np.random.randint(0, 10000, (32, 50)), requires_grad=False)

# Create padding mask
mask = create_padding_mask(sequences, pad_idx=0)

# Forward pass
output = model(sequences, mask=mask)
print(output.shape)  # (32, 10)
"""
