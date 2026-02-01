#!/usr/bin/env python3
"""
Module that calculates the cumulative n-gram BLEU score
"""
import math
from collections import Counter


def make_ngrams(words, n):
    """
    Generates a list of n-grams from a list of words

    Args:
        words (list of str): list of words
        n (int): n-gram size

    Returns:
        list of tuples: n-grams
    """
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence

    Args:
        references (list of list of str): reference translations
        sentence (list of str): candidate sentence
        n (int): largest n-gram to use

    Returns:
        float: cumulative n-gram BLEU score
    """
    precisions = []

    # Step 1: Compute precision for each k-gram (1..n)
    for k in range(1, n+1):
        cand_ngrams = make_ngrams(sentence, k)
        cand_counts = Counter(cand_ngrams)

        max_ref_counts = {}
        for ref in references:
            ref_ngrams = Counter(make_ngrams(ref, k))
            for ng in cand_counts:
                if ng in ref_ngrams:
                    max_ref_counts[ng] = max(max_ref_counts.get(ng, 0),
                                             ref_ngrams[ng])
                else:
                    max_ref_counts[ng] = max_ref_counts.get(ng, 0)

        clipped_counts = {ng: min(cand_counts[ng],
                                  max_ref_counts.get(ng, 0))
                          for ng in cand_counts}

        clipped_sum = sum(clipped_counts.values())
        total_cand = sum(cand_counts.values())
        precision = clipped_sum / total_cand if total_cand > 0 else 0.0
        precisions.append(precision)

    # Step 2: Compute geometric mean of precisions
    if min(precisions) == 0:
        geo_mean = 0.0
    else:
        log_sum = sum(math.log(p) for p in precisions)
        geo_mean = math.exp(log_sum / n)

    # Step 3: Compute brevity penalty (BP)
    c = len(sentence)
    ref_lengths = [len(ref) for ref in references]
    r = min(ref_lengths, key=lambda ref_len: (abs(ref_len - c), ref_len))
    bp = 1.0 if c > r else math.exp(1 - r / c) if c != 0 else 0.0

    # Step 4: Compute cumulative BLEU
    bleu = bp * geo_mean
    return bleu
