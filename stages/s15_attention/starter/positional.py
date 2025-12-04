from __future__ import annotations

import numpy as np


def sinusoidal_encoding(max_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_length: Maximum sequence length
        d_model: Model dimension

    Returns:
        encodings: Shape (max_length, d_model)
    """
    # TODO:
    # pe = np.zeros((max_length, d_model))
    # position = np.arange(max_length)[:, np.newaxis]
    # div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    #
    # pe[:, 0::2] = np.sin(position * div_term)
    # pe[:, 1::2] = np.cos(position * div_term)
    #
    # return pe
    raise NotImplementedError


def add_positional_encoding(x: np.ndarray, pe: np.ndarray) -> np.ndarray:
    """
    Add positional encoding to input embeddings.

    Args:
        x: Input embeddings, shape (batch, seq_len, d_model)
        pe: Positional encodings, shape (max_len, d_model)

    Returns:
        output: x + pe[:seq_len]
    """
    # TODO:
    # seq_len = x.shape[1]
    # return x + pe[:seq_len]
    raise NotImplementedError


def learned_positional_encoding(max_length: int, d_model: int) -> np.ndarray:
    """
    Initialize learned positional embeddings.

    Args:
        max_length: Maximum sequence length
        d_model: Model dimension

    Returns:
        embeddings: Shape (max_length, d_model)
    """
    # TODO:
    # return np.random.randn(max_length, d_model) * 0.02
    raise NotImplementedError


def create_causal_mask(seq_length: int) -> np.ndarray:
    """
    Create causal (look-ahead) mask.

    Args:
        seq_length: Sequence length

    Returns:
        mask: Lower triangular boolean matrix
    """
    # TODO:
    # return np.tril(np.ones((seq_length, seq_length), dtype=bool))
    raise NotImplementedError


def create_padding_mask(lengths: np.ndarray, max_length: int) -> np.ndarray:
    """
    Create padding mask from sequence lengths.

    Args:
        lengths: Actual sequence lengths, shape (batch,)
        max_length: Maximum sequence length

    Returns:
        mask: Shape (batch, max_length)
    """
    # TODO:
    # batch = len(lengths)
    # positions = np.arange(max_length)[np.newaxis, :]
    # lengths_expanded = lengths[:, np.newaxis]
    # return positions < lengths_expanded
    raise NotImplementedError
