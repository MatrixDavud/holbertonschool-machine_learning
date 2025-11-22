#!/usr/bin/env python3
"""Implementing Multivariate Probabiity concepts."""
import numpy as np


def correlation(C):
    """Calculate the correlation matrix from a covariance matrix."""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std = np.sqrt(np.diag(C))
    denom = np.outer(std, std)
    R = C / denom
    np.fill_diagonal(R, 1.0)

    return R
