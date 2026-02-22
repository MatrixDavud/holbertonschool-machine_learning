#!/usr/bin/env python3
"""
Module that performs semantic search on a corpus of documents.
"""

import os
import tensorflow as tf
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    Args:
        corpus_path (str): path to corpus directory
        sentence (str): query sentence

    Returns:
        str: reference text of the most similar document
    """
    if not corpus_path or not sentence:
        return None

    documents = []
    texts = []

    for filename in os.listdir(corpus_path):
        path = os.path.join(corpus_path, filename)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                text = f.read()
                documents.append(text)
                texts.append(text)

    if not texts:
        return None

    model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder/4"
    )

    embeddings = model(texts)
    query_embedding = model([sentence])

    similarity = tf.keras.losses.cosine_similarity(
        query_embedding,
        embeddings
    )

    best_index = tf.argmax(-similarity).numpy()

    return documents[best_index]
