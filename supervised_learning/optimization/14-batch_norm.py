#!/usr/bin/env python3
"""Implementing optimizations."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Create a batch normalization layer for a neural network."""
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'),
        use_bias=False
    )(prev)

    bn = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        gamma_initializer='ones',
        beta_initializer='zeros'
    )(dense, training=True)

    output = tf.keras.layers.Activation(activation)(bn)

    return output
