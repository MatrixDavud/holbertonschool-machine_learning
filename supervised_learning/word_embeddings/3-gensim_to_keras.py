#!/usr/bin/env python3
"""
Module to convert a gensim Word2Vec
model to a Keras Embedding layer
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer.

    Args:
        model: A trained gensim word2vec model

    Returns:
        The trainable keras Embedding layer
    """
    # Copy embeddings to avoid modifying gensim weights
    embedding_matrix = model.wv.vectors.copy()

    vocab_size, embedding_dim = embedding_matrix.shape

    layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True
    )

    return layer
