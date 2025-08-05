#!/usr/bin/env python3
"""Performing some math operations."""


def matrix_shape(matrix):
    """Calculate the dimensions of a matrix."""
    shape = []
    shape.append(len(matrix))
    check_list = matrix[0]
    while type(check_list) is list:
        shape.append(len(check_list))
        check_list = check_list[0]
    return shape
