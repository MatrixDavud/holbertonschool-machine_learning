#!/usr/bin/env python3
"""Clustering Algorithms impelementation."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d)
        k (int): Number of clusters
        iterations (int): Maximum number of iterations

    Returns:
        C, clss or (None, None) on failure
        C (numpy.ndarray): Centroids, shape (k, d)
        clss (numpy.ndarray): Cluster index for each point, shape (n,)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    C = np.random.uniform(min_vals, max_vals, size=(k, d))

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

        clss = np.argmin(distances, axis=1)

        old_C = C.copy()

        for i in range(k):
            points = X[clss == i]
            if points.shape[0] == 0:
                C[i] = np.random.uniform(min_vals, max_vals, size=(d,))
            else:
                C[i] = np.mean(points, axis=0)
        if np.allclose(C, old_C):
            break

    return C, clss
