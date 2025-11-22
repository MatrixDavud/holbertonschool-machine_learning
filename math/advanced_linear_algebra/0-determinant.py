#!/usr/bin/env python3
"""Implementing Advanced Linear Algebra concepts."""


def determinant(matrix):
    """Calculate the determinant of a matrix."""
    if not isinstance(matrix, list) or not\
            all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    if n == 0:
        return 1
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        minor = [row[:col] + row[col+1:] for row in matrix[1:]]

        sign = (-1) ** col

        det += sign * matrix[0][col] * determinant(minor)

    return det
