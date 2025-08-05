#!/usr/bin/env python3
"""Perform operations on ndarray."""


def np_elementwise(mat1, mat2):
    """Return elementwise add,sub,mul,div as tuple."""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
