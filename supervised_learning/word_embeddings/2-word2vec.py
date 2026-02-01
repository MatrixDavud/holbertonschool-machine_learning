#!/usr/bin/env python3
"""
Word2Vec model training using gensim.
"""

import re
from gensim.models import Word2Vec


def word2vec_model(
    sentences,
    vector_size=100,
    min_count=5,
    window=5,
    negative=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1
):
    """
    Creates, builds, and trains a Word2Vec model.

    Args:
        sentences (list): List of sentences to train on.
        vector_size (int): Dimensionality of word vectors.
        min_count (int): Minimum word frequency.
        window (int): Maximum distance between words.
        negative (int): Number of negative samples.
        cbow (bool): True for CBOW, False for Skip-gram.
        epochs (int): Number of training epochs.
        seed (int): Random seed.
        workers (int): Number of worker threads.

    Returns:
        Word2Vec: Trained Word2Vec model.
    """
    if not isinstance(sentences, list):
        raise TypeError("sentences must be a list")

    tokenized = []
    for sentence in sentences:
        if not isinstance(sentence, str):
            raise TypeError("each sentence must be a string")

        clean = sentence.lower()
        clean = re.sub(r"[^\w\s]", "", clean)
        tokens = clean.split()
        tokenized.append(tokens)

    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0 if cbow else 1,
        negative=negative,
        seed=seed,
        workers=workers
    )

    model.train(
        tokenized,
        total_examples=len(tokenized),
        epochs=epochs
    )

    return model
