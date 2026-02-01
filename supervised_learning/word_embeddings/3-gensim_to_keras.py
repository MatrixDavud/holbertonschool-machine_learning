#!/usr/bin/env python3
"""
Module for converting a gensim word2vec model to a Keras Embedding layer.
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer.

    Args:
        model: A trained gensim word2vec model

    Returns:
        A trainable keras Embedding layer with weights from the word2vec
        model
    """
    # Get the word vectors from the model
    word_vectors = model.wv

    # Get the weight matrix from word vectors
    weights = word_vectors.vectors

    # Get vocabulary size and vector dimensions
    vocab_size, vector_size = weights.shape

    # Create a Keras Embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )

    return embedding_layer
