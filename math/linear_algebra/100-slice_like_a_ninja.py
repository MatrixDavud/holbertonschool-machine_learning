#!/usr/bin/env python3
"""Perform operations on ndarray."""


def np_slice(matrix, axes={}):
    """Slice a matrix along specific axes using the axes dictionary."""
    slices = [slice(None)] * matrix.ndim
    for axis, sl in axes.items():
        slices[axis] = slice(*sl)
    return matrix[tuple(slices)]
