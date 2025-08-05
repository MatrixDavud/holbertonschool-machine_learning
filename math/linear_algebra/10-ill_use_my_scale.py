#!/usr/bin/env python 3
"""Perform ndarray operations."""


def np_shape(matrix):
    """Return the shape of matrix."""
    shape = []
    shape.append(len(matrix))
    check_list = matrix[0]
    while type(check_list) is list:
        shape.append(len(check_list))
        check_list = check_list[0]
    return tuple(shape)
