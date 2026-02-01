#!/usr/bin/env python3
"""
Module for creating bag of words embedding matrix.
"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences: A list of sentences to analyze
        vocab: A list of the vocabulary words to use for the analysis
               If None, all words within sentences should be used

    Returns:
        embeddings: A numpy.ndarray of shape (s, f) containing the
                    embeddings where s is the number of sentences in
                    sentences and f is the number of features analyzed
        features: A list of the features used for embeddings
    """
    # Tokenize all sentences
    tokenized_sentences = []
    for sentence in sentences:
        # Convert to lowercase and split into words
        words = sentence.lower().split()
        tokenized_sentences.append(words)

    # Build vocabulary if not provided
    if vocab is None:
        # Collect all unique words from sentences
        vocab_set = set()
        for words in tokenized_sentences:
            vocab_set.update(words)
        # Sort vocabulary for consistent ordering
        features = sorted(list(vocab_set))
    else:
        # Use provided vocabulary (convert to lowercase)
        features = [word.lower() for word in vocab]

    # Create word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(features)}

    # Initialize embeddings matrix
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Fill embeddings matrix with word counts
    for i, words in enumerate(tokenized_sentences):
        for word in words:
            if word in word_to_idx:
                embeddings[i, word_to_idx[word]] += 1

    return embeddings, features
