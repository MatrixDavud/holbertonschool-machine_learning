#!/usr/bin/env python3
"""Poisson Distribution."""
e = 2.7182818285


class Poisson:
    """Poisson distribution class."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson distribution object."""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    @staticmethod
    def factorial(n):
        """Calculates the factorial of n iteratively."""
        if not isinstance(n, int) or n < 0:
            raise ValueError("Input must be a non-negative integer.")
        if n == 0:
            return 1
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    def pmf(self, k):
        """Calculate the value of the PMF for a given number of successes."""
        if k < 0:
            return 0
        else:
            k = int(k)
        return (e**-self.lambtha * self.lambtha**k)/self.factorial(k)

    def cdf(self, k):
        """Calculate the value of the CDF for a given number of successes."""
        if k < 0:
            return 0
        else:
            k = int(k)
        cdf_sum = 0
        for i in range(k+1):
            cdf_sum += (e**-self.lambtha * self.lambtha**i)/self.factorial(i)
        return cdf_sum
