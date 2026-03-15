#!/usr/bin/env python3
"""
Module to determine the next action using epsilon-greedy
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    Args:
        Q: a numpy.ndarray containing the q-table
        state: the current state
        epsilon: the epsilon to use for the calculation

    Returns:
        the next action index
    """
    # Generate a random number to decide between explore and exploit
    p = np.random.uniform(0, 1)

    if p < epsilon:
        # Explore: pick a random action index
        # Q.shape[1] gives the number of actions
        action = np.random.randint(0, Q.shape[1])
    else:
        # Exploit: pick the action with the highest Q-value for the state
        action = np.argmax(Q[state])

    return action
