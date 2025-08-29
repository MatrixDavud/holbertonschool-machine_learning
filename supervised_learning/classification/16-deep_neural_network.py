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

        layers_arr = np.array(layers)

        if not np.issubdtype(layers_arr.dtype, np.integer) or np.any(layers_arr <= 0):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            # number of nodes in current layer
            nodes = layers[l - 1]

            # number of nodes in the previous layer (nx for first layer)
            prev_nodes = nx if l == 1 else layers[l - 2]

            # He initialization for weights
            self.weights["W{}".format(l)] = (
                np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            )

            # Bias initialized as zeros
            self.weights["b{}".format(l)] = np.zeros((nodes, 1))
