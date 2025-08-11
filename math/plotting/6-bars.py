#!/usr/bin/env python3
"""Bar graph."""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Create a bar graph representing number of fruit per person."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

