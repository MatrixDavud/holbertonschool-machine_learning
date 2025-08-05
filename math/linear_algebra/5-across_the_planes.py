#!/usr/bin/env python3
"""Performing some linear algebra."""


def add_matrices2D(mat1, mat2):
    """Return elementwise addition of matrices,None if non-matching shapes."""
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        sum_matrix = [[mat1[row][column] + mat2[row][column] for column in
                       range(len((mat1[0])))] for row in range(len(mat1))]
        return sum_matrix
    else:
        return None
