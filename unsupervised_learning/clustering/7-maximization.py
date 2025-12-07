#!/usr/bin/env python3
"""Gaussian Mixture Model implementation."""
import numpy as np


def maximization(X, g):
    """Calculate the maximization step in the EM algorithm for a GMM.
    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        g (numpy.ndarray): Posterior probabilities for each data point
        in each cluster of shape (k, n).
        Returns: pi, m, S or (None, None, None) on failure.
            - pi is a numpy.ndarray of shape (k,) containing the updated
            priors for each cluster.
            - m is a numpy.ndarray of shape (k, d) containing the updated
            centroid means for each cluster.
            - S is a numpy.ndarray of shape (k, d, d) containing the
            updated covariance matrices for each cluster."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n_g != n:
        return None, None, None

    if np.any(g < 0):
        return None, None, None

    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None
    N_k = np.sum(g, axis=1)

    if np.any(N_k == 0):
        return None, None, None

    pi = N_k / n

    m = (g @ X) / N_k[:, None]

    S = np.zeros((k, d, d))

    for i in range(k):
        diff = X - m[i]
        weighted_diff = diff.T * g[i]
        S[i] = weighted_diff @ diff / N_k[i]

    return pi, m, S
