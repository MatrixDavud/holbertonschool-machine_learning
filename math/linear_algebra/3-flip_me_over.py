#!/usr/bin/env python3
"""Performing some matrix operations."""


def matrix_transpose(matrix):
    """Return the transpose of the 2D matrix."""
    transpose = [[matrix[i][j] for i in range(len(matrix))]
                 for j in range(len(matrix[0]))]
    return transpose
