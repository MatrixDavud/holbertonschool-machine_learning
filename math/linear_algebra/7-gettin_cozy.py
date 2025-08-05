#!/usr/bin/env python3
"""Performing Matrix operations."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate matrices in given axis."""
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        res_mat = [row[:] for row in mat1] + [row[:] for row in mat2]
        return res_mat
    elif axis == 1 and len(mat1) == len(mat2):
        res_mat = [mat1[i] + mat2[i] for i in range(len(mat2))]
        return res_mat
    else:
        return None
