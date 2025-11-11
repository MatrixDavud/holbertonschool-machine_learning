#!/usr/bin/env python3
"""Binomial Distribution."""


class Binomial:
    """Binomial distribution class."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize Binomial distribution object."""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = 0
            for i in data:
                variance += (i - mean)**2
            variance = variance / len(data)
            p = 1 - variance / mean
            n = mean / p
            n = round(n)
            p = mean / n
            self.n = n
            self.p = p
