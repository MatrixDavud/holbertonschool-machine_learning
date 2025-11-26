#!/usr/bin/env python3
"""Implementing Bayesian Probability concepts."""
import numpy as np


def likelihood(x, n, P):
    """Calculate the likelihood of obtaining the data."""
    if n < 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int and x < 0:
        raise ValueError("x must be an integer that is\
                         greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")