#!/usr/bin/env python3
"""Performing some linear algebra."""


def add_arrays(arr1, arr2):
    """Return element wise addition of arrays,None if non-matching shape."""
    if len(arr1) != len(arr2):
        return None
    else:
        sum_arr = []
        for i in range(len(arr1)):
            sum_arr.append(arr1[i]+arr2[i])
        return sum_arr
