#!/usr/bin/env python3
"""Building a model with keras library."""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Convert a label vector into a one-hot matrix."""
    return K.utils.to_categorical(labels, num_classes=classes)
