#!/usr/bin/env python3
"""Applying regularixation methods to the model."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determine if Gradient Descent should be stopped early."""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    stop = count >= patience
    return stop, count
