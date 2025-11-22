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


def minor(matrix):
    """Calculate the minor matrix of a matrix."""
    if not isinstance(matrix, list) or not\
            all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or any(len(row) == 0 for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minor_matrix = []

    for i in range(n):
        row_minors = []
        for j in range(n):
            submatrix = [
                row[:j] + row[j+1:]
                for k, row in enumerate(matrix)
                if k != i
            ]

            row_minors.append(determinant(submatrix))
        minor_matrix.append(row_minors)

    return minor_matrix


def cofactor(matrix):
    """Calculate the cofactor matrix of a matrix."""
    if not isinstance(matrix, list) or not \
            all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or any(len(row) == 0 for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    minors = minor(matrix)

    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            sign = (-1) ** (i + j)
            row_cofactors.append(sign * minors[i][j])
        cofactor_matrix.append(row_cofactors)

    return cofactor_matrix


def adjugate(matrix):
    """Calculate the adjugate matrix of a matrix."""
    if not isinstance(matrix, list) or not \
            all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or any(len(row) == 0 for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    cof = cofactor(matrix)

    adj = []
    for i in range(n):
        new_row = []
        for j in range(n):
            new_row.append(cof[j][i])
        adj.append(new_row)

    return adj
