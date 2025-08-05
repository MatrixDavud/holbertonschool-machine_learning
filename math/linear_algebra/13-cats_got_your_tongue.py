#!/usr/bin/env python3
"""Perform operations on ndarray."""
import numpy as np

def np_cat(mat1, mat2, axis=0):
    """Concatenate two matrices along the given axis."""
    return np.concatenate((mat1, mat2), axis)
