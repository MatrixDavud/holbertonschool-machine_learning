#!/usr/bin/env python3
"""
Bag of Words embedding implementation.
"""

import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences (list): List of sentences to analyze.
        vocab (list, optional): Vocabulary words to use.
            If None, vocabulary is built from sentences.

    Returns:
        tuple: (embeddings, features)
            embeddings is a numpy.ndarray of shape (s, f)
            features is a list of vocabulary words
    """
    if not isinstance(sentences, list):
        raise TypeError("sentences must be a list")

    tokenized = []
    for sentence in sentences:
        if not isinstance(sentence, str):
            raise TypeError("each sentence must be a string")
        tokens = sentence.lower().split()
        tokenized.append(tokens)

    if vocab is None:
        vocab_set = set()
        for tokens in tokenized:
            vocab_set.update(tokens)
        features = sorted(vocab_set)
    else:
        features = list(vocab)

    s = len(sentences)
    f = len(features)

    embeddings = np.zeros((s, f), dtype=int)

    word_to_index = {}
    for idx, word in enumerate(features):
        word_to_index[word] = idx

    for i, tokens in enumerate(tokenized):
        for word in tokens:
            if word in word_to_index:
                j = word_to_index[word]
                embeddings[i, j] += 1

    return embeddings, features
