#!/usr/bin/env python3
"""Calculus operations."""


def poly_derivative(poly):
    """Return the list of coefs of the dy of the polynomial."""
    if type(poly) is not list or not \
            all(isinstance(coef, (int, float)) for coef in poly):
        return None

    if len(poly) == 1:
        return [0]

    derivative = [poly[i] * i for i in range(1, len(poly))]

    if all(coef == 0 for coef in derivative):
        return [0]

    return derivative
