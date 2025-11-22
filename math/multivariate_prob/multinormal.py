#!/usr/bin/env python3
"""Implementing Multivariate Probabiity concepts."""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""
    
    def __init__(self, data):
        """
        data: numpy.ndarray of shape (d, n)
        d = number of dimensions
        n = number of data points
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        data_centered = data - self.mean

        self.cov = (data_centered @ data_centered.T) / (n - 1)
