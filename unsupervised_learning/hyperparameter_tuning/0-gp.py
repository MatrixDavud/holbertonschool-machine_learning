#!/usr/bin/env python3
"""Gaussian Process Implementation."""
import numpy as np


class GaussianProcess:
    """Gaussian Process class."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Construct object."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Calculate the covariance kernel matrix between two matrices."""
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) \
            + np.sum(X2**2, axis=1) \
            - 2 * np.dot(X1, X2.T)

        return (self.sigma_f ** 2) * np.exp(-0.5 / self.l**2 * sqdist)
