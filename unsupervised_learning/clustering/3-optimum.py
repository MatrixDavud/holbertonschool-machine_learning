#!/usr/bin/env python3
"""Clustering Algorithms implementation."""
import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Args:
        X (numpy.ndarray): Data set of shape (n, d)
        kmin (int): Minimum number of clusters to evaluate
        kmax (int): Maximum number of clusters to evaluate
        iterations (int): Max iterations for K-means

    Returns:
        results: list of (C, clss) for each k
        d_vars: list of variance differences from smallest k

        Or (None, None) on failure.
    """
    kmeans = __import__('1-kmeans').kmeans
    variance = __import__('2-variance').variance


    if kmax - kmin < 1:
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None

        results.append((C, clss))
        variances.append(variance(X, C))

    base = variances[0]
    d_vars = [base - v for v in variances]

    return results, d_vars
