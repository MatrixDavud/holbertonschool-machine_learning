#!/usr/bin/env python3
"""One hot decoding implementation."""
import numpy as np


def one_hot_decode(one_hot):
    """Convert a one-hot matrix into a vector of labels."""
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    try:
        decoded = np.argmax(one_hot, axis=0)
        return decoded
    except Exception:
        return None
