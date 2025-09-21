#!/usr/bin/env python3
"""Applying regularixation methods to the model."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Update the Ws and bs of a NN using GD with L2 regularization."""
    m = Y.shape[1]

    A_prev_key = "A{}".format(L)
    A_L = cache[A_prev_key]
    dZ = A_L - Y

    for i in range(L, 0, -1):
        W_key = "W{}".format(i)
        b_key = "b{}".format(i)
        A_prev_key = "A{}".format(i - 1)

        Wl = weights[W_key]
        bl = weights[b_key]
        A_prev = cache[A_prev_key]

        dW = (1.0 / m) * np.dot(dZ, A_prev.T) + (lambtha / m) * Wl
        db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA_prev = np.dot(Wl.T, dZ)

        weights[W_key] = Wl - alpha * dW
        weights[b_key] = bl - alpha * db

        if i > 1:
            A_prev_activation = A_prev
            dZ = dA_prev * (1 - A_prev_activation ** 2)
