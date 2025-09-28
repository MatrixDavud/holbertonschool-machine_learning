#!/usr/bin/env python3
"""Performing convolutions on images."""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Perform a valid convolution on grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    conv_h = h-kh+1
    conv_w = w-kw+1
    convolved_images = np.zeros((m, conv_h, conv_w))
    for i in range(conv_h):
        for k in range(conv_w):
            convolved_images[:, i, k] = np.sum(images[:, i:i+kh, k:k+kw]
                                               * kernel, axis=(1, 2))
    return convolved_images
