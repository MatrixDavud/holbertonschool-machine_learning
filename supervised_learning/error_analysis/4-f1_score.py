#!/usr/bin/env python3
"""Conducting error analysis in our model."""
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculate the F1 score of a confusion matrix."""
    sens = sensitivity(confusion)
    prec = precision(confusion)
    f1 = 2 * (sens * prec) / (sens + prec)
    return f1
