#!/usr/bin/env python3
"""Clustering Algorithms impelementation."""
import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means algorithm.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters (positive integer).

    Returns:
        numpy.ndarray|None: Array of shape (k, d) with initialized
        centroids, or ''None'' on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    low = X.min(axis=0)
    high = X.max(axis=0)

    try:
        centroids = np.random.uniform(low=low, high=high, size=(k, d))
    except Exception:
        return None

    return centroids
