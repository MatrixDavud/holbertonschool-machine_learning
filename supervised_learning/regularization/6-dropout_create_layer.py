#!/usr/bin/env python3
"""Applying regularixation methods to the model."""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Create a layer of a neural network using dropout."""
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_avg"
        )
    )(prev)

    if training and keep_prob < 1.0:
        layer = tf.keras.layers.Dropout(rate=1 -
                                        keep_prob)(layer, training=training)

    return layer
