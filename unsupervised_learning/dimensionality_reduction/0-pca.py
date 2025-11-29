#!/usr/bin/env python3
"""Dimensionality Reduction algorithms implementations."""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on dataset X and returns the weights matrix W.

    X: numpy.ndarray (n, d)
       Data whose mean across each dimension is already 0.
    var: float
       Fraction of variance to retain.

    Returns:
        W: numpy.ndarray of shape (d, nd)
           PCA weights matrix
    """

    # Compute covariance matrix
    cov = np.cov(X, rowvar=False)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Compute cumulative explained variance
    cumvar = np.cumsum(eigvals) / np.sum(eigvals)

    # Number of components required
    k = np.searchsorted(cumvar, var) + 1

    # Return *weights matrix*, not projection
    W = eigvecs[:, :k]

    return W
