#!/usr/bin/env python3
"""Classification algorithm using Deep Neural Network (DNN class)."""
import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network class."""

    def __init__(self, nx, layers):
        """Construct the deep neural network object."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0\
                or not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.__L + 1):
            # number of nodes in current layer
            nodes = layers[l - 1]

            # number of nodes in the previous layer (nx for first layer)
            prev_nodes = nx if l == 1 else layers[l - 2]

            # He initialization for weights
            self.__weights["W{}".format(l)] = (
                np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            )

            # Bias initialized as zeros
            self.__weights["b{}".format(l)] = np.zeros((nodes, 1))

    @property
    def L(self):
        """Get the value of length of layers."""
        return self.__L

    @property
    def cache(self):
        """Get the value of cache."""
        return self.__cache

    @property
    def weights(self):
        """Get the value of the weights."""
        return self.__weights
