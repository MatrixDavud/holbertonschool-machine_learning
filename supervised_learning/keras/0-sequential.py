#!/usr/bin/env python3
"""Building a model with keras library."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a neural network with the Keras library."""
    model = K.Sequential()

    model.add(
        K.layers.Dense(
            layers[0],
            activation=activations[0],
            kernel_regularizer=K.regularizers.l2(lambtha),
            input_shape=(nx,)
        )
    )

    if keep_prob < 1 and len(layers) > 1:
        model.add(K.layers.Dropout(1 - keep_prob))

    for i in range(1, len(layers)):
        model.add(
            K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
            )
        )
        if keep_prob < 1 and i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
