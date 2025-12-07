#!/usr/bin/env python3
"""Gaussian Mixture Model implementation."""
import numpy as np


def initialize(X, k):
    """Initialize variables for a Gaussian Mixture Model.
    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters (positive integer).
        Returns: (pi, m, S), or (None, None, None) on failure.
            - pi is a numpy.ndarray of shape (k,) containing the priors for
            each cluster, initialized evenly.
            m is a numpy.ndarray of shape (k, d) containing the centroid
            means for each cluster, initialized with K-means.
            - S is a numpy.ndarray of shape (k, d, d) containing the
            covariance matrices for each cluster, initialized as identity
            matrices."""
    kmeans = __import__('1-kmeans').kmeans

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None
    n, d = X.shape
    if k > n:
        return None, None, None

    pi = np.full(k, 1 / k)

    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
