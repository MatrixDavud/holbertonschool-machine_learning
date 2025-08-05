#!/usr/bin/env python3
"""Performing some linear algebra."""


def mat_mul(mat1, mat2):
    """Return the result matrix after mul,None if it isn't possible."""
    a, b, c, d = len(mat1), len(mat1[0]), len(mat2), len(mat2[0])
    if b != c:
        return None
    else:
        res_mat = [[sum(mat1[i][k] * mat2[k][j] for k in range(b))
                    for j in range(d)] for i in range(a)]
        return res_mat
