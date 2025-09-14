#!/usr/bin/env python3
"""Implementing optimizations."""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Update a variable in place using the Adam optimization algorithm."""
    v_new = beta1 * v + (1 - beta1) * grad

    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    v_corrected = v_new / (1 - beta1 ** t)
    s_corrected = s_new / (1 - beta2 ** t)

    var_updated = var - alpha * v_corrected / ((s_corrected) ** 0.5 + epsilon)

    return var_updated, v_new, s_new
