#!/usr/bin/env python3
"""Prediction module"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Make a prediction using a neural network."""
    predictions = network.predict(data, verbose=verbose)
    return predictions
