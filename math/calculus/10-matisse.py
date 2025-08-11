#!/usr/bin/env python3
"""Calculus operations."""


def poly_derivative(poly):
    """Return the list of coefs of the dy of the polynomial."""
    if type(poly) is not list or\
            not all(isinstance(coef, (int, float)) for coef in poly):
        return None
    else:
        if len(poly) == 1:
            dy = [0]
            return dy
        else:
            if all(coef == 0 for coef in poly):
                return [0]
            dy_list = []
            for i in range(1, len(poly)):
                dy_list.append(poly[i]*i)
            return dy_list
