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

    def forward_prop(self, X):
        """Calculate the forward propagation of neural network."""
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression."""
        cost = np.sum((Y*np.log(A) + (1 - Y)*np.log(1.0000001 - A)))
        cost = cost / -Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions."""
        self.forward_prop(X)
        p = self.__A2
        cost = self.cost(Y, p)
        labels = np.where(p >= 0.5, 1, 0)
        return labels, cost
