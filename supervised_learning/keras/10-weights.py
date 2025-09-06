#!/usr/bin/env python3
"""Saves and loads the model weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Save the model's weights."""
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """Load the model's weights."""
    network.load_weights(filename)
