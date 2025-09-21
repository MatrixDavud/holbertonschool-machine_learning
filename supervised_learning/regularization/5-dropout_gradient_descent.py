#!/usr/bin/env python3
"""Applying regularixation methods to the model."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Update the weights of a NN with Dropout regularization using GD."""
    m = Y.shape[1]
    dZ = {}

    for i in reversed(range(1, L + 1)):
        A = cache['A' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]

        if i == L:
            dZ[i] = A - Y
        else:
            dA = np.dot(weights['W' + str(i + 1)].T, dZ[i + 1])

            D = cache['D' + str(i)]
            dA *= D
            dA /= keep_prob

            dZ[i] = dA * (1 - A ** 2)

        dW = np.dot(dZ[i], A_prev.T) / m
        db = np.sum(dZ[i], axis=1, keepdims=True) / m

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
