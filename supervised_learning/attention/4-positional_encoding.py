#!/usr/bin/env python3
"""
Positional encoding for Transformer models
"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer

    Args:
        max_seq_len (int): maximum sequence length
        dm (int): model depth

    Returns:
        np.ndarray: positional encoding of shape (max_seq_len, dm)
    """
    # Initialize positional encoding matrix
    PE = np.zeros((max_seq_len, dm))

    # Position indices (max_seq_len, 1)
    positions = np.arange(max_seq_len)[:, np.newaxis]

    # Dimension indices (1, dm)
    dims = np.arange(dm)[np.newaxis, :]

    # Compute angle rates
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / dm)
    angles = positions * angle_rates

    # Apply sin to even indices, cos to odd indices
    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])

    return PE
