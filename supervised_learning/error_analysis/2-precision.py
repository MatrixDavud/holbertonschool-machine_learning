#!/usr/bin/env python3
"""Conducting error analysis in our model."""
import numpy as np


def precision(confusion):
    """Calculate the precision for each class in a confusion matrix."""
    m = confusion.shape[0]
    precision_classes = np.zeros(m)
    for i in range(m):
        prec = confusion[i][i] / np.sum(confusion[:, i])
        precision_classes[i] = prec
    return precision_classes
