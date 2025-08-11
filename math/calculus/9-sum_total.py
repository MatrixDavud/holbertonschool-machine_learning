#!/usr/bin/env python3
"""Math operations."""


def summation_i_squared(n):
    """Return the summation of i^2 till n, where n should be valid int."""
    if type(n) is not int:
        return None
    else:
        return int((n * (n+1) * (2*n+1))/6)
