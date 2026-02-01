#!/usr/bin/env python3
"""
Module that calculates the n-gram BLEU score for a candidate sentence
"""
import math
from collections import Counter


def make_ngrams(words, n):
    """
    Generates a list of n-grams from a list of words.

    Args:
        words (list of str): list of words
        n (int): size of n-grams

    Returns:
        list of tuples: n-grams
    """
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence

    Args:
        references (list of list of str): reference translations
        sentence (list of str): candidate sentence
        n (int): n-gram size

    Returns:
        float: n-gram BLEU score
    """
    # Step 1: Create n-grams for candidate
    cand_ngrams = make_ngrams(sentence, n)
    cand_counts = Counter(cand_ngrams)

    # Step 2: Create n-grams for references and get max counts
    max_ref_counts = {}
    for ref in references:
        ref_ngrams = Counter(make_ngrams(ref, n))
        for ng in cand_counts:
            if ng in ref_ngrams:
                max_ref_counts[ng] = max(max_ref_counts.get(ng, 0),
                                         ref_ngrams[ng])
            else:
                max_ref_counts[ng] = max_ref_counts.get(ng, 0)

    # Step 3: Clip candidate counts
    clipped_counts = {ng: min(cand_counts[ng], max_ref_counts.get(ng, 0))
                      for ng in cand_counts}

    # Step 4: Calculate precision
    clipped_sum = sum(clipped_counts.values())
    total_cand = sum(cand_counts.values())
    precision = clipped_sum / total_cand if total_cand > 0 else 0.0

    # Step 5: Compute brevity penalty (BP)
    c = len(sentence)
    ref_lengths = [len(ref) for ref in references]
    r = min(ref_lengths, key=lambda ref_len: (abs(ref_len - c), ref_len))
    bp = 1.0 if c > r else math.exp(1 - r / c) if c != 0 else 0.0

    # Step 6: Compute BLEU
    bleu = bp * precision
    return bleu
