#!/usr/bin/env python3
"""Saves and loads to json or from json to model configs."""
import tensorflow.keras as K


def save_config(network, filename):
    """Save the model's configuration in JSON format."""
    config_json = network.to_json()
    with open(filename, "w") as f:
        f.write(config_json)


def load_config(filename):
    """Load the model from json format."""
    with open(filename, "r") as f:
        config_json = f.read()
    model = K.models.model_from_json(config_json)
    return model
