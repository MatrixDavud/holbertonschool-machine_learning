#!/usr/bin/env python3
"""RNN forward propagation module."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Perform forward propagation for a simple RNN over a sequence.

    Args:
        rnn_cell: Instance of RNNCell used to perform each time step.
        X (np.ndarray): Input data of shape (t, m, i).
        h_0 (np.ndarray): Initial hidden state of shape (m, h).

    Returns:
        tuple: (H, Y)
            H (np.ndarray): All hidden states of shape (t + 1, m, h).
            Y (np.ndarray): All outputs of shape (t, m, o).
    """
    t, m, _ = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
