#!/usr/bin/env python3
"""
Module for creating and training a word2vec model using gensim.
"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds and trains a gensim word2vec model.

    Args:
        sentences: A list of sentences to be trained on
        vector_size: The dimensionality of the embedding layer
        min_count: The minimum number of occurrences of a word for use
                   in training
        window: The maximum distance between the current and predicted
                word within a sentence
        negative: The size of negative sampling
        cbow: A boolean to determine the training type; True is for CBOW;
              False is for Skip-gram
        epochs: The number of iterations to train over
        seed: The seed for the random number generator
        workers: The number of worker threads to train the model

    Returns:
        The trained word2vec model
    """
    # Determine the training algorithm
    # sg=0 for CBOW, sg=1 for Skip-gram
    sg = 0 if cbow else 1

    # Create and build the Word2Vec model
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )
    
    # Build vocabulary from sentences
    model.build_vocab(sentences)
    
    # Train the model
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
