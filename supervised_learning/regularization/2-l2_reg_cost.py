#!/usr/bin/env python3
"""Applying regularixation methods to the model."""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculate the cost of a neural network with L2 regularization."""
    reg_loss = tf.add_n(model.losses)
    total_cost = cost + reg_loss

    return total_cost
