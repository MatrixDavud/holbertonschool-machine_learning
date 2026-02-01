#!/usr/bin/env python3
"""
TF-IDF embedding implementation.
"""

import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

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

        clean = sentence.lower()
        clean = re.sub(r"[^\w\s]", "", clean)
        tokens = clean.split()
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

    embeddings = np.zeros((s, f), dtype=float)

    word_to_index = {}
    for idx, word in enumerate(features):
        word_to_index[word] = idx

    # Document frequency
    df = np.zeros(f, dtype=int)
    for tokens in tokenized:
        seen = set(tokens)
        for word in seen:
            if word in word_to_index:
                df[word_to_index[word]] += 1

    # Inverse Document Frequency
    idf = np.log(s / (1 + df))

    # TF-IDF computation
    for i, tokens in enumerate(tokenized):
        total_words = len(tokens)
        for word in tokens:
            if word in word_to_index:
                j = word_to_index[word]
                tf = tokens.count(word) / total_words
                embeddings[i, j] = tf * idf[j]

    return embeddings, features
