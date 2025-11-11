#!/usr/bin/env python3
"""Poisson Distribution."""


class Poisson:
    """Poisson distribution class."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson object."""
        if not data:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if type(data) is list and len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)
