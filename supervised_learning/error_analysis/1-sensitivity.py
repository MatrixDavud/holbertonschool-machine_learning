#!/usr/bin/env python3
"""Conducting error analysis in our model."""
import numpy as np


def sensitivity(confusion):
    """Calculate the sensitivity for each class in a confusion matrix."""
    m = confusion.shape[0]
    sensitivity_classes = np.zeros(m)
    for i in range(m):
        sens = confusion[i][i] / np.sum(confusion[i])
        sensitivity_classes[i] = sens
    return sensitivity_classes
