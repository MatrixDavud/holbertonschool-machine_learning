#!/usr/bin/env python3
"""One hot encoding implementation."""
import numpy as np


def one_hot_encode(Y, classes):
    """Convert numeric label vector into a one-hot matrix."""
    try:
        m = Y.shape[0]
        ohe_matrix = np.zeros((classes, m))
        ohe_matrix[Y, np.arange(m)] = 1
        return ohe_matrix
    except Exception:
        return None
