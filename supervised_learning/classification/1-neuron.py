#!/usr/bin/env python3
"""Classification algorithm implementation."""
import numpy as np


class Neuron:
    """Neuron class."""

    def __init__(self, nx):
        """Construct the neuron object."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Get the value of the weight."""
        return self.__W

    @property
    def b(self):
        """Get the value of bias."""
        return self.__b

    @property
    def A(self):
        """Get the value of prediction."""
        return self.__A
