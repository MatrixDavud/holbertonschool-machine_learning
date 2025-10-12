#!/usr/bin/env python3
"""Applying regularixation methods to the model."""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conduct forward propagation using Dropout."""
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        Z = np.dot(W, A_prev) + b

        if i == L:
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = expZ / np.sum(expZ, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)

            D = (np.random.rand(*A.shape) < keep_prob).astype(int)
            A *= D
            A /= keep_prob
            cache['D' + str(i)] = D

        cache['A' + str(i)] = A

    return cache

# (3,4), np.random.rand(3,4)
# # #