#!/usr/bin/env python3
"""Perform matrix operations."""


def matrix_shape(matrix):
    """Calculate the dimensions of a matrix."""
    shape = []
    shape.append(len(matrix))
    check_list = matrix[0]
    while type(check_list) is list:
        shape.append(len(check_list))
        check_list = check_list[0]
    return shape


def cat_matrices(mat1, mat2, axis=0):
    """Check shape and concatenation compatibility."""
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)

    # Check shape compatibility for concat:
    if len(shape1) != len(shape2):
        return None
    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None
    return r_cat_matrices(mat1, mat2, axis)


def r_cat_matrices(mat1, mat2, axis=0):  # recursive function to concatenate
    """Concatenate recursively."""
    if axis == 0:
        return mat1 + mat2
    else:
        # Concatenate recursively along axis-1
        return [r_cat_matrices(mat1[i], mat2[i], axis=axis-1)
                for i in range(len(mat1))]
