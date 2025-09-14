#!/usr/bin/env python3
"""Implementing optimizations."""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Create a learning rate decay operation using inverse time decay."""
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return lr_schedule
