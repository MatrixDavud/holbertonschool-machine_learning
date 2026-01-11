#!/usr/bin/env python3
"""
3-generate_faces.py

This module defines convolutional_GenDiscr(), a helper that builds a
convolutional generator and a convolutional discriminator (critic/classifier)
suitable for 16x16 grayscale face images.

The architectures are designed to match the project specification:
- Generator input shape: (16,)
- Generator output shape: (16, 16, 1)
- Discriminator input shape: (16, 16, 1)
- Uses "tanh" activations (also after Conv2D blocks)
- Conv2D layers use padding="same"
"""

from tensorflow import keras


def convolutional_GenDiscr():
    """
    Build and return a convolutional generator and discriminator.

    Returns
    -------
    tuple
        (generator, discriminator), two tf.keras.Model objects.
    """
    def get_generator():
        """Create the convolutional generator model."""
        inputs = keras.Input(shape=(16,))
        x = keras.layers.Dense(2048, activation="tanh")(inputs)

        x = keras.layers.Reshape((2, 2, 512))(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(
            16,
            kernel_size=(3, 3),
            padding="same",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(
            1,
            kernel_size=(3, 3),
            padding="same",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation("tanh")(x)

        return keras.Model(inputs, outputs, name="generator")

    def get_discriminator():
        """Create the convolutional discriminator model."""
        inputs = keras.Input(shape=(16, 16, 1))

        x = keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same",
        )(inputs)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
        )(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
        )(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(
            256,
            kernel_size=(3, 3),
            padding="same",
        )(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1)(x)

        return keras.Model(inputs, outputs, name="discriminator")

    return get_generator(), get_discriminator()
