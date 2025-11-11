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

    @staticmethod
    def factorial(n):
        """Calculate the factorial of n iteratively."""
        if not isinstance(n, int) or n < 0:
            raise ValueError("Input must be a non-negative integer.")
        if n == 0:
            return 1
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    def n_choose_k(self, n, k):
        """Calculate n choose k."""
        return self.factorial(n) / (self.factorial(k)*self.factorial(n-k))

    def pmf(self, k):
        """Calculate the value of the PMF for a given number of successes."""
        if k < 0 or k > self.n:
            return 0
        else:
            k = int(k)
            factor_1 = self.n_choose_k(self.n, k)
            factor_2 = self.p**k * (1-self.p)**(self.n - k)
            return factor_1 * factor_2

    def cdf(self, k):
        """Calculate the value of the CDF for a given number of successes."""
        if k < 0 or k > self.n:
            return 0
        else:
            k = int(k)
            cdf = 0
            for i in range(k+1):
                cdf += self.pmf(i)
            return cdf
