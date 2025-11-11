#!/usr/bin/env python3
"""Normal Distribution."""


class Normal:
    """Normal distribution class."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal Distribution object."""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
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
