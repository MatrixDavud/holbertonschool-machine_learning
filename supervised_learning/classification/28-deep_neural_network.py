#!/usr/bin/env python3
"""Classification algorithm using Deep Neural Network (DNN class) for multiclass classification."""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Deep Neural Network class for multiclass classification."""

    def __init__(self, nx, layers, activation='sig'):
        """Construct the deep neural network object."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            nodes = layers[i]
            prev_nodes = nx if i == 0 else layers[i - 1]

            self.__weights["W{}".format(i + 1)] = (
                np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            self.__weights["b{}".format(i + 1)] = np.zeros((nodes, 1))

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

    @property
    def activation(self):
        """Get the value of the activation function."""
        return self.__activation

    def forward_prop(self, X):
        """Calculate forward propagation of the neural network."""
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W{}'.format(i)]
            A = self.__cache['A{}'.format(i-1)]
            b = self.__weights['b{}'.format(i)]
            z = np.dot(W, A) + b
            
            # Use softmax for the output layer, specified activation for hidden layers
            if i == self.__L:
                # Softmax activation for output layer (multiclass)
                exp_z = np.exp(z)
                self.__cache['A{}'.format(i)] = exp_z / np.sum(exp_z, axis=0, keepdims=True)
            else:
                # Use specified activation function for hidden layers
                if self.__activation == 'sig':
                    self.__cache['A{}'.format(i)] = 1 / (1 + np.exp(-z))
                elif self.__activation == 'tanh':
                    self.__cache['A{}'.format(i)] = np.tanh(z)

        return self.__cache['A{}'.format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression."""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-15)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions."""
        self.forward_prop(X)
        A = self.__cache['A{}'.format(self.__L)]
        cost = self.cost(Y, A)
        
        # Convert predictions to class labels (one-hot format)
        predictions = np.zeros_like(A)
        max_indices = np.argmax(A, axis=0)
        predictions[max_indices, np.arange(A.shape[1])] = 1
        
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network."""
        m = Y.shape[1]
        AL = cache['A{}'.format(self.__L)]
        dZl = AL - Y
        for i in range(self.__L, 0, -1):
            Al = cache['A{}'.format(i-1)]
            dwl = (dZl @ Al.T) / m
            dbl = (np.sum(dZl, axis=1, keepdims=True)) / m

            Al_prev = cache['A{}'.format(i-1)]
            Wl = self.__weights['W{}'.format(i)]
            if i > 1:
                # Calculate dZl for previous layer based on activation function
                if self.__activation == 'sig':
                    dZl = (Wl.T @ dZl) * (Al_prev * (1-Al_prev))
                elif self.__activation == 'tanh':
                    dZl = (Wl.T @ dZl) * (1 - Al_prev * Al_prev)
            self.__weights['W{}'.format(i)] -= alpha * dwl
            self.__weights['b{}'.format(i)] -= alpha * dbl

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Train the deep neural network."""
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
            cache_l, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

            if (iteration % step == 0) or (iteration == iterations):
                cost = self.cost(Y, cache_l)
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

    def save(self, filename):
        """Save the instance object to a file in pickle format."""
        try:
            if not filename.endswith(".pkl"):
                filename += ".pkl"
            with open(filename, "wb") as file:
                pickle.dump(self, file)
        except Exception:
            return None

    @staticmethod
    def load(filename):
        """Load a pickled DeepNeuralNetwork object."""
        try:
            with open(filename, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
