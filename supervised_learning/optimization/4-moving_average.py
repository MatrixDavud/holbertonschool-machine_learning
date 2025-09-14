#!/usr/bin/env python3
"""Implementing optimizations."""


def moving_average(data, beta):
    """Calculate the weighted moving average of a data set."""
    moving_averages = []
    v = 0

    for t, x in enumerate(data, start=1):
        v = beta * v + (1 - beta) * x

        v_corrected = v / (1 - beta**t)

        moving_averages.append(v_corrected)

    return moving_averages
