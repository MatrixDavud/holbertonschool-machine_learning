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
    import re

    # Tokenize all sentences
    tokenized_sentences = []
    for sentence in sentences:
        # Convert to lowercase, extract only alphanumeric words
        # Filter out single character 's' from possessives
        words = [w for w in re.findall(r'\b\w+\b', sentence.lower())
                 if len(w) > 1 or w != 's']
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
        # Use provided vocabulary (convert to lowercase and clean)
        features = []
        for word in vocab:
            cleaned_words = [w for w in re.findall(r'\b\w+\b',
                             word.lower()) if len(w) > 1 or w != 's']
            if cleaned_words:
                features.append(cleaned_words[0])

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

    return embeddings, np.array(features)
