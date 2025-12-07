#!/usr/bin/env python3
"""Clustering Algorithms impelementation."""
import numpy as np


def variance(X, C):
    """Calculate the total intra-cluster variance for a data set.
    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        C (numpy.ndarray): Centroids of shape (k, d).
        Returns: var, or None on failure.
            - var is the total intra-cluster."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    distances = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)

    closest = np.min(distances, axis=1)

    return np.sum(closest ** 2)
