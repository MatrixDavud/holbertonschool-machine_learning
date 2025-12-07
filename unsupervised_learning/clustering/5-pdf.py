#!/usr/bin/env python3
"""Gaussian Mixture Model implementation."""
import numpy as np


def pdf(X, m, S):
    """Calculate the PMF of a Gaussian distribution.
    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        m (numpy.ndarray): Mean of shape (d,).
        S (numpy.ndarray): Covariance matrix of shape (d, d).
        Returns: P or None on failure.
            - P is a numpy.ndarray of shape (n,) containing the PDF
            values for each data point in X."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    n, d = X.shape
    if m.shape[0] != d:
        return None
    if S.shape != (d, d):
        return None

    try:
        det_S = np.linalg.det(S)
        if det_S <= 0:
            return None

        inv_S = np.linalg.inv(S)
    except (np.linalg.LinAlgError, ValueError):
        return None

    diff = X - m

    quad = np.sum((diff @ inv_S) * diff, axis=1)

    norm_const = np.sqrt(((2 * np.pi) ** d) * det_S)

    P = np.exp(-0.5 * quad) / norm_const

    P = np.maximum(P, 1e-300)

    return P
