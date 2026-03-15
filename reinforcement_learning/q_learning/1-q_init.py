#!/usr/bin/env python3
"""
Module to initialize a Q-table for reinforcement learning
"""
import numpy as np


def q_init(env):
    """
    Initializes the Q-table as a numpy.ndarray of zeros

    Args:
        env: the FrozenLakeEnv instance

    Returns:
        the Q-table as a numpy.ndarray of zeros
    """
    # Number of states (rows) and number of actions (columns)
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    # Initialize the Q-table with zeros
    q_table = np.zeros((state_space_size, action_space_size))

    return q_table
