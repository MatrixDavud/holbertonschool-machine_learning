#!/usr/bin/env python3
"""Implementing optimizations."""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalize an unactivated output of a NN using batch normalization."""
    mu = np.mean(Z, axis=0, keepdims=True)

    var = np.mean((Z - mu) ** 2, axis=0, keepdims=True)

    Z_norm = (Z - mu) / np.sqrt(var + epsilon)

    Z_tilde = gamma * Z_norm + beta

    return Z_tilde
