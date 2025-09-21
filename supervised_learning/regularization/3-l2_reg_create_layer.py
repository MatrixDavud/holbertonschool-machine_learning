#!/usr/bin/env python3
"""Applying regularixation methods to the model."""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Create a neural network layer in tensorFlow with L2 regularization."""
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_avg"
        ),
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    )(prev)

    return layer
