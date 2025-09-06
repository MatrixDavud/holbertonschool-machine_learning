#!/usr/bin/env python3
"""Testing module."""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Test the neural network."""
    loss, accuracy = network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )
    return [loss, accuracy]
