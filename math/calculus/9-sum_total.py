#!/usr/bin/env python3
"""Math operations."""


def summation_i_squared(n):
    """Return the summation of i^2 till n, where n should be valid int."""
    if type(n) is not int:
        return None
    else:
        sum = 0
        for i in range(n+1):
            sum += i ** 2
        return sum
