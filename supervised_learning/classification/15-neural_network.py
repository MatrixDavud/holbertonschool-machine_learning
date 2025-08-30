#!/usr/bin/env python3
"""Classification algorithm using neural network (NN class)."""
import numpy as np
import matplotlib as plt


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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network."""

        m = X.shape[1]

        # Output layer gradients
        dZ2 = A2 - Y
        dW2 = (1/m) * np.matmul(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        # Hidden layer gradients
        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (1/m) * np.matmul(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # Update
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Train the neural network."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if graph or verbose:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")


        costs, iteration_list = [], []

        for iteration in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

            if (iteration % step == 0) or (iteration == iterations):
                cost = self.cost(Y, A2)
                if verbose:
                    print(f"Cost after {iteration} iterations: {cost}")
                if graph:
                    costs.append(cost)
                    iteration_list.append(iteration)

        if graph:
            plt.plot(iteration_list, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training cost")
            plt.show()

        self.forward_prop(X)
        return self.evaluate(X, Y)
