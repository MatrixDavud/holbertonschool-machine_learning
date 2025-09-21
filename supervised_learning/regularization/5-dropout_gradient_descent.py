#!/usr/bin/env python3
"""Applying regularixation methods to the model."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Update the weights of a NN with Dropout regularization using GD."""
    m = Y.shape[1]
    
    # Start backpropagation from the output layer
    dZ = cache['A' + str(L)] - Y
    
    # Backpropagate through all layers
    for l in range(L, 0, -1):
        # Get the previous layer's activation
        if l == 1:
            A_prev = cache['A0']
        else:
            A_prev = cache['A' + str(l - 1)]
        
        # Calculate dA for the previous layer FIRST (before updating weights)
        if l > 1:
            dA_prev = np.dot(weights['W' + str(l)].T, dZ)
            # Apply dropout mask and scaling
            dA_prev *= cache['D' + str(l - 1)]
            dA_prev /= keep_prob
        
        # Calculate gradients for current layer
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Update weights and biases
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db
        
        # Calculate dZ for the previous layer
        if l > 1:
            # For tanh activation: dZ = dA * (1 - A^2)
            A_prev_layer = cache['A' + str(l - 1)]
            dZ = dA_prev * (1 - A_prev_layer ** 2)
