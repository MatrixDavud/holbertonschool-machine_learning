#!/usr/bin/env python3
"""Classification algorithm implementation (only Neuron class)."""
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

    def forward_prop(self, X):
        """Calculate forward propagation of the neuron."""
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """Return the cost of the model using logistic regression."""
        # sum from i = 1 to N, y_i * log(A) + (1 - y_i) * log(1.0000001 - A)
        cost = np.sum((Y*np.log(A) + (1 - Y)*np.log(1.0000001 - A)))
        cost = cost / -Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neuron's predictions."""
        z = np.dot(self.__W, X) + self.__b
        p = 1 / (1 + np.exp(-z))
        cost = np.sum((Y*np.log(p) + (1 - Y)*np.log(1.0000001 - p)))
        cost = cost / -Y.shape[1]
        labels = np.where(p >= 0.5, 1, 0)
        return labels, cost
