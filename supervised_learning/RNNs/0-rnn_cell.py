#!/usr/bin/env python3
"""RNN cell module."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """
        Initialize the RNN cell.

        Args:
            i (int): Dimensionality of the input data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the output.
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev (np.ndarray): Previous hidden state of shape (m, h).
            x_t (np.ndarray): Input data for time step t of shape (m, i).

        Returns:
            tuple: (h_next, y)
                h_next (np.ndarray): Next hidden state of shape (m, h).
                y (np.ndarray): Output of shape (m, o).
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        logits = np.matmul(h_next, self.Wy) + self.by
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)
        return h_next, y
