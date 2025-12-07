#!/usr/bin/env python3
"""Clustering Algorithms implementation."""
import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Test for the optimum number of clusters by variance.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        kmin (int): Minimum number of clusters to check (inclusive).
        kmax (int): Maximum number of clusters to check (inclusive).
        iterations (int): Maximum number of iterations for K-means.

    Returns:
        tuple: (results, d_vars) where:
            - results is a list of K-means outputs for each cluster size
            - d_vars is a list of variance differences from smallest cluster
        Returns (None, None) on failure.
    """
    kmeans = __import__('1-kmeans').kmeans
    variance = __import__('2-variance').variance

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    n, d = X.shape

    if not isinstance(kmin, int) or kmin <= 0 or kmin >= n:
        return None, None

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0 or kmax > n:
        return None, None

    if kmin >= kmax:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)

        if C is None or clss is None:
            return None, None

        results.append((C, clss))

        var = variance(X, C)

        if var is None:
            return None, None

        variances.append(var)

    baseline_variance = variances[0]
    d_vars = [baseline_variance - var for var in variances]

    return results, d_vars
