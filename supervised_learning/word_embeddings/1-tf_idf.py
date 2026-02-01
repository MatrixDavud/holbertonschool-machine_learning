#!/usr/bin/env python3
"""
Module for creating TF-IDF embedding matrix.
"""
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    TF-IDF (Term Frequency-Inverse Document Frequency) is calculated as:
    TF-IDF(t, d) = TF(t, d) * IDF(t)

    Where:
    - TF(t, d) = (Number of times term t appears in document d) /
                 (Total number of terms in document d)
    - IDF(t) = log((Total number of documents) /
                   (Number of documents containing term t))

    Args:
        sentences: A list of sentences to analyze
        vocab: A list of the vocabulary words to use for the analysis
               If None, all words within sentences should be used

    Returns:
        embeddings: A numpy.ndarray of shape (s, f) containing the
                    TF-IDF embeddings where s is the number of sentences
                    and f is the number of features analyzed
        features: A numpy array of the features used for embeddings
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

    # Initialize matrices
    s = len(sentences)
    f = len(features)
    tf_matrix = np.zeros((s, f))

    # Calculate Term Frequency (TF)
    for i, words in enumerate(tokenized_sentences):
        doc_length = len(words)
        if doc_length > 0:
            for word in words:
                if word in word_to_idx:
                    tf_matrix[i, word_to_idx[word]] += 1
            # Normalize by document length to get TF
            tf_matrix[i] = tf_matrix[i] / doc_length

    # Calculate Inverse Document Frequency (IDF)
    idf_vector = np.zeros(f)
    for j, feature in enumerate(features):
        # Count how many documents contain this feature
        doc_count = 0
        for words in tokenized_sentences:
            if feature in words:
                doc_count += 1
        # Calculate IDF
        if doc_count > 0:
            idf_vector[j] = np.log(s / doc_count)
        else:
            idf_vector[j] = 0

    # Calculate TF-IDF by multiplying TF with IDF
    embeddings = tf_matrix * idf_vector

    return embeddings, np.array(features)
