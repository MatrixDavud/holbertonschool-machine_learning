#!/usr/bin/env python3
"""Classification algorithm using neural network (NN class)."""
import numpy as np


class NeuralNetwork:
    """Neural Network class."""

    def __init__(self, nx, nodes):
        """Construct the neural network object."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.nx = nx
        self.nodes = nodes

        # Hidden layer
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Output neuron
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Get the value of the weight."""
        return self.__W1

    @property
    def b1(self):
        """Get the value of bias."""
        return self.__b1

    @property
    def A1(self):
        """Get the value of prediction."""
        return self.__A1

    @property
    def W2(self):
        """Get the value of the weight."""
        return self.__W2

    @property
    def b2(self):
        """Get the value of bias."""
        return self.__b2

    @property
    def A2(self):
        """Get the value of prediction."""
        return self.__A2
