#!/usr/bin/env python3
"""Implementing optimizations."""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Update the learning rate using inverse time decay."""
    step_count = np.floor(global_step / decay_step)

    alpha_new = alpha / (1 + decay_rate * step_count)

    return alpha_new
