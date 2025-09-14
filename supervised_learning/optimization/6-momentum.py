#!/usr/bin/env python3
"""Implementing optimizations."""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Set G.D with momentum alg, using TF."""
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
