#!/usr/bin/env python3
"""Creating a Deep Convolutional Neural Network."""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Build a projection block."""
    F11, F3, F12 = filters
    he_normal = K.initializers.HeNormal(seed=0)

    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(s, s),
                        padding='valid', kernel_initializer=he_normal)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', kernel_initializer=he_normal)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(1, 1),
                        padding='valid', kernel_initializer=he_normal)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    shortcut = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                               strides=(s, s), padding='valid',
                               kernel_initializer=he_normal)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
