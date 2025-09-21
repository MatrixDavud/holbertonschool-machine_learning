#!/usr/bin/env python3
"""Applying regularixation methods to the model."""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculate the cost of a neural network with L2 regularization."""
    sum_squared = 0.0
    for i in range(1, L + 1):
        W = weights.get(f"W{i}")
        sum_squared += np.sum(np.square(W))

    reg_term = (lambtha / (2.0 * m)) * sum_squared

    return cost + reg_term
