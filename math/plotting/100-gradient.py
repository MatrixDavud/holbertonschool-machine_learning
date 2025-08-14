#!/usr/bin/env python3
"""Plotting advanced."""
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """Plot the mountain elevation graph."""
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    scatter_plot = plt.scatter(x, y, c=z, cmap='viridis')
    plt.colorbar(scatter_plot, label='elevation (m)')
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Mountain Elevation')
    plt.show()
