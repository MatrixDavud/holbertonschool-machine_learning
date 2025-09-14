#!/usr/bin/env python3
"""Implementing optimizations."""


def normalize(X, m, s):
    """Normalize (standardizes) a matrix."""
    X_norm = (X - m) / s
    return X_norm
