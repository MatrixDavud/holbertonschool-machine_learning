#!/usr/bin/env python3
"""Applying regularixation methods to the model."""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculate the cost of a neural network with L2 regularization."""
    reg_losses = model.losses

    reg_losses_tensor = tf.stack(reg_losses)
    total_costs = cost + reg_losses_tensor

    return total_costs
