#!/usr/bin/env python3
"""Dimensionality Reduction algorithms implementations."""
import numpy as np


def pca(X, var=0.95):
    """Perform PCA on dataset X.

    Args:
        X: numpy.ndarray of shape (n, d) where n is the number of data points
           and d is the number of dimensions.
        var: float, the fraction of variance to retain.

    Returns:
        X_reduced: numpy.ndarray of shape (n, k) where k is the number of
                   dimensions after reduction.
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute covariance matrix
    covariance_matrix = np.cov(X_centered, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Compute cumulative variance ratio
    cumulative_variance = np.cumsum(sorted_eigenvalues)
    total_variance = cumulative_variance[-1]
    variance_ratio = cumulative_variance / total_variance

    # Determine number of components to retain
    k = np.searchsorted(variance_ratio, var) + 1

    # Project data onto the top k eigenvectors
    X_reduced = np.dot(X_centered, sorted_eigenvectors[:, :k])

    return X_reduced
