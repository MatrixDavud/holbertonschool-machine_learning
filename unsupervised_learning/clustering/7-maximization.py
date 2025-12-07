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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    if np.any(g < 0) or np.any(g > 1):
        return None, None, None

    pi = np.sum(g, axis=1) / n

    n_k = np.sum(g, axis=1, keepdims=True)
    m = (g @ X) / n_k

    S = np.zeros((k, d, d))

    for cluster in range(k):

        X_centered = X - m[cluster]

        weighted_X = X_centered * g[cluster, :, np.newaxis]

        S[cluster] = (X_centered.T @ weighted_X) / n_k[cluster, 0]

    return pi, m, S
