#!/usr/bin/env python3
"""Implementing optimizations."""
import numpy as np


def normalization_constants(X):
    """Calculate the normalization (standardization) constants of a matrix."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
