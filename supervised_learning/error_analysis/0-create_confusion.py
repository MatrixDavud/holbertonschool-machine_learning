#!/usr/bin/env python3
"""Conducting error analysis in our model."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Create a confusion matrix of the model."""
    n = labels.shape[1]
    conf_matrix = np.zeros((n, n))
    for i in range(labels.shape[0]):
        true_label = np.argmax(labels[i])
        pred_label = np.argmax(logits[i])
        conf_matrix[true_label, pred_label] += 1
    return conf_matrix
