#!/usr/bin/env python3
"""Implementing optimizations."""
import numpy as np


def shuffle_data(X, Y):
    """Shuffle the data points in two matrices the same way."""
    m = X.shape[0]
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    return X_shuffled, Y_shuffled
