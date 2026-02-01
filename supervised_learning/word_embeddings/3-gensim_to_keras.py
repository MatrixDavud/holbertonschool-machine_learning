#!/usr/bin/env python3
"""
Convert a gensim Word2Vec model to a Keras Embedding layer.
"""

import tensorflow.keras as keras


def gensim_to_keras(model):
    """
    Converts a gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model (gensim.models.Word2Vec): Trained Word2Vec model.

    Returns:
        keras.layers.Embedding: Trainable embedding layer.
    """
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape

    embedding = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )

    return embedding
