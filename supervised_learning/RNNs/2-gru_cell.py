#!/usr/bin/env python3
"""Module for implementing a Gated Recurrent Unit (GRU) cell."""
import numpy as np


class GRUCell:
    """
    Represents a Gated Recurrent Unit (GRU) cell.

    A GRU is a type of recurrent neural network that uses gating
    mechanisms to control information flow, helping to solve the
    vanishing gradient problem in traditional RNNs.
    """

    def __init__(self, i, h, o):
        """
        Initialize the GRU cell.

        Args:
            i: Dimensionality of the input data
            h: Dimensionality of the hidden state
            o: Dimensionality of the outputs

        The GRU uses three gates:
        - Update gate (z): Controls how much of the previous hidden
          state to keep
        - Reset gate (r): Controls how much of the previous hidden
          state to forget when computing candidate hidden state
        - Candidate hidden state (h_candidate): New memory content
        """
        # Update gate weights: combines input and previous hidden state
        # Shape: (i + h, h) - concatenated input contributes to h units
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))

        # Reset gate weights: determines what to forget from prev state
        # Shape: (i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))

        # Intermediate (candidate) hidden state weights
        # Shape: (i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))

        # Output weights: projects hidden state to output space
        # Shape: (h, o)
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the
                    previous hidden state
            x_t: numpy.ndarray of shape (m, i) containing the data
                 input for the cell

        Returns:
            h_next: The next hidden state of shape (m, h)
            y: The output of the cell of shape (m, o)

        GRU Forward Pass Theory:
        1. Concatenate input x_t and previous hidden state h_prev
        2. Update gate: z_t = sigmoid([x_t, h_prev] @ Wz + bz)
           - Decides how much to update the hidden state
        3. Reset gate: r_t = sigmoid([x_t, h_prev] @ Wr + br)
           - Decides how much past information to forget
        4. Candidate hidden state:
           h_candidate = tanh([x_t, r_t * h_prev] @ Wh + bh)
           - Compute new candidate values for hidden state
        5. Final hidden state:
           h_next = z_t * h_prev + (1 - z_t) * h_candidate
           - Mix old and new hidden states based on update gate
        6. Output: y = softmax(h_next @ Wy + by)
        """
        # Step 1: Concatenate input and previous hidden state
        # Shape: (m, i) + (m, h) -> (m, i + h)
        concat_xh = np.concatenate((h_prev, x_t), axis=1)

        # Step 2: Compute update gate (z_t)
        # Sigmoid squashes values to [0, 1]
        # z_t close to 1 means keep old state, close to 0 means update
        z_t = self._sigmoid(np.matmul(concat_xh, self.Wz) + self.bz)

        # Step 3: Compute reset gate (r_t)
        # Determines how much of previous hidden state to use when
        # computing candidate hidden state
        r_t = self._sigmoid(np.matmul(concat_xh, self.Wr) + self.br)

        # Step 4: Compute candidate hidden state
        # Reset gate applied element-wise to previous hidden state
        # This allows the model to drop irrelevant information
        concat_x_rh = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_candidate = np.tanh(
            np.matmul(concat_x_rh, self.Wh) + self.bh
        )

        # Step 5: Compute next hidden state
        # Linear interpolation between previous state and candidate
        # z_t acts as a learned "mixing coefficient"
        h_next = z_t * h_prev + (1 - z_t) * h_candidate

        # Step 6: Compute output using softmax activation
        # Projects hidden state to output dimension
        logits = np.matmul(h_next, self.Wy) + self.by
        y = self._softmax(logits)

        return h_next, y

    def _sigmoid(self, x):
        """
        Compute the sigmoid activation function.

        Args:
            x: Input array

        Returns:
            Sigmoid of x, element-wise
        """
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        """
        Compute the softmax activation function.

        Args:
            x: Input array of shape (m, o)

        Returns:
            Softmax probabilities of shape (m, o)

        Numerically stable implementation that subtracts max value
        to prevent overflow.
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
