#!/usr/bin/env python3
"""Exponential Distribution."""
e = 2.7182818285


class Exponential:
    """Exponential distribution class."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Exponential distribution object."""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        """Calculate the value of the PDF for a given time period."""
        if x < 0:
            return 0
        else:
            return self.lambtha * e**(-self.lambtha * x)

    def cdf(self, x):
        """Calculate the value of the CDF for a given time period."""
        if x < 0:
            return 0
        else:
            return 1 - e**(-self.lambtha * x)
