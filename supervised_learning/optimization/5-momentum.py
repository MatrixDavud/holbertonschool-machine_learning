#!/usr/bin/env python3
"""Implementing optimizations."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update a variable using G.D. with momentum optimization algorithm."""
    v_new = beta1 * v + (1 - beta1) * grad

    var_updated = var - alpha * v_new

    return var_updated, v_new
