#!/usr/bin/env python3
"""Performing Calculus."""


def poly_integral(poly, C=0):
    """Return the list of coefs of the anti-dy of polynomial."""
    if type(poly) is not list:
        return None

    if not (isinstance(C, int) or isinstance(C, float)):
        return None

    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None
    if len(poly) == 0:
        return None
    coefs = []
    coefs.append(C)
    for i in range(len(poly)):
        coefs.append(poly[i] / (i+1))
    coefs = [int(c) if isinstance(c, float) and
             c.is_integer() else c for c in coefs]

    while len(coefs) > 1 and coefs[-1] == 0:
        coefs.pop()
    return coefs
