#!/usr/bin/env python3
"""Savind and loading models."""
import tensorflow.keras as K


def save_model(network, filename):
    """Save the entire model."""
    network.save(filename)


def load_model(filename):
    """Load the entire model from filename."""
    return K.models.load_model(filename)
