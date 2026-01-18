#!/usr/bin/env python3
"""GRU cell module."""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit (GRU) cell."""

    def __init__(self, i, h, o):
        """
        Initialize the GRU cell.

        Args:
            i (int): Dimensionality of the input data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the output.
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """Compute the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Compute softmax with numerical stability."""
        x_shift = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x_shift)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev (np.ndarray): Previous hidden state of shape (m, h).
            x_t (np.ndarray): Input data at time step t of shape (m, i).

        Returns:
            tuple: (h_next, y)
                h_next (np.ndarray): Next hidden state of shape (m, h).
                y (np.ndarray): Output of shape (m, o).
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        z_gate = self.sigmoid(np.matmul(concat, self.Wz) + self.bz)
        r_gate = self.sigmoid(np.matmul(concat, self.Wr) + self.br)

        r_hidden = r_gate * h_prev
        concat_candidate = np.concatenate((r_hidden, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_candidate, self.Wh) + self.bh)

        h_next = (1 - z_gate) * h_prev + z_gate * h_tilde

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
