#!/usr/bin/env python3
"""Applying regularixation methods to the model."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Update the weights of a NN with Dropout regularization using GD."""
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        if i == 1:
            A_prev = cache['A0']
        else:
            A_prev = cache['A' + str(i - 1)]

        if i > 1:
            dA_prev = np.dot(weights['W' + str(i)].T, dZ)
            dA_prev *= cache['D' + str(i - 1)]
            dA_prev /= keep_prob

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

        if i > 1:
            A_prev_layer = cache['A' + str(i - 1)]
            dZ = dA_prev * (1 - A_prev_layer ** 2)
