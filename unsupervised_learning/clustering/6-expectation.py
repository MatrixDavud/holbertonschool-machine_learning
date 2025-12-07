#!/usr/bin/env python3
"""Gaussian Mixture Model implementation."""
import numpy as np


def expectation(X, pi, m, S):
    """Calculate the expectation step in the EM algorithm for a GMM.
    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        pi (numpy.ndarray): Priors for each cluster of shape (k,).
        m (numpy.ndarray): Centroid means for each cluster of shape (k, d).
        S (numpy.ndarray): Covariance matrices for each cluster of shape
        (k, d, d).
        Returns: g, l or (None, None) on failure.
            - g is a numpy.ndarray of shape (k, n) containing the
            posterior probabilities for each data point in each cluster.
            - l is the total log likelihood."""
    pdf = __import__('5-pdf').pdf

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d):
        return None, None
    if S.shape != (k, d, d):
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        g[i] = pi[i] * P

    total_P = np.sum(g, axis=0)
    if np.any(total_P == 0):
        return None, None

    g /= total_P

    log_likelihood = np.sum(np.log(total_P))

    return g, log_likelihood
