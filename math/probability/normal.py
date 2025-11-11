#!/usr/bin/env python3
"""Normal Distribution."""
pi = 3.1415926536
e = 2.7182818285


class Normal:
    """Normal distribution class."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal distribution object."""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            sum_stddev = 0
            for i in data:
                sum_stddev += (i - self.mean)**2
            sum_stddev = (sum_stddev / len(data)) ** 0.5
            self.stddev = sum_stddev

    def z_score(self, x):
        """Calculate the z-score of a given x-value."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate the x-value of a given z-score."""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculate the value of the PDF for a given x-value."""
        factor = 1 / (self.stddev * (2*pi)**0.5)
        exponent = e ** (-0.5*(self.z_score(x))**2)
        return factor * exponent

    @staticmethod
    def error_function(x):
        return (2/(pi)**0.5)*(x - x**3/3 + x**5/10 - x**7/42 + x**9/216)

    def cdf(self, x):
        """Calculate the value of the CDF for a given x-value."""
        return 0.5*(1 + self.error_function(self.z_score(x)/(2**0.5)))
