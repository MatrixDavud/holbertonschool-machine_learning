#!/usr/bin/env python3
"""
Module that calculates the unigram BLEU score for a candidate sentence
"""
import math
from collections import Counter


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence

    Args:
        references (list of list of str): reference translations
        sentence (list of str): candidate sentence

    Returns:
        float: unigram BLEU score
    """
    # Step 1: Count unigrams in candidate sentence
    cand_counts = Counter(sentence)

    # Step 2: Get max counts from references
    max_ref_counts = {}
    for ref in references:
        ref_counts = Counter(ref)
        for word in cand_counts:
            if word in ref_counts:
                max_ref_counts[word] = max(max_ref_counts.get(word, 0),
                                           ref_counts[word])
            else:
                max_ref_counts[word] = max_ref_counts.get(word, 0)

    # Step 3: Clip candidate counts
    clipped_counts = {word: min(cand_counts[word], max_ref_counts.get(word, 0))
                      for word in cand_counts}

    # Step 4: Calculate precision
    clipped_sum = sum(clipped_counts.values())
    total_cand = sum(cand_counts.values())
    precision = clipped_sum / total_cand if total_cand > 0 else 0.0

    # Step 5: Compute brevity penalty (BP)
    c = len(sentence)
    ref_lengths = [len(ref) for ref in references]
    # Choose reference length closest to candidate length
    r = min(ref_lengths, key=lambda ref_len: (abs(ref_len - c), ref_len))
    bp = 1.0 if c > r else math.exp(1 - r / c) if c != 0 else 0.0

    # Step 6: Compute BLEU
    bleu = bp * precision
    return bleu
