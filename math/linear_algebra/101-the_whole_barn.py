#!/usr/bin/env python3
"""Performing matrix operations."""


def matrix_shape(matrix):
    """Calculate the dimensions of a matrix."""
    shape = []
    shape.append(len(matrix))
    check_list = matrix[0]
    while type(check_list) is list:
        shape.append(len(check_list))
        check_list = check_list[0]
    return shape


def add_matrices(mat1, mat2):
    """Return none if the shapes aren't equal,recursive function otherwise."""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    return _add_matrices_recursive(mat1, mat2)


def _add_matrices_recursive(m1, m2):
    """Add matrix elements recursively."""
    if not isinstance(m1, list):
        return m1 + m2
    return [_add_matrices_recursive(a, b) for a, b in zip(m1, m2)]
