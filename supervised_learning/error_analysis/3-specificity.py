#!/usr/bin/env python3
"""Conducting error analysis in our model."""
import numpy as np


def specificity(confusion):
    """Calculate the specificity for each class in a confusion matrix."""
    m = confusion.shape[0]
    specificity_classes = np.zeros(m)
    for k in range(m):
        tp = confusion[k, k]
        fn = np.sum(confusion[k, :]) - confusion[k, k]
        fp = np.sum(confusion[:, k]) - confusion[k, k]
        tn = np.sum(confusion) - (tp + fp + fn)
        spec = tn / (tn + fp)
        specificity_classes[k] = spec
    return specificity_classes
