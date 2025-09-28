#!/usr/bin/env python3
"""Performing convolutions on images."""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Perform a same convolution on grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    if h != h-kh+1 and w != w-kw+1:
        p_h = int(np.ceil((kh-1)/2))
        p_w = int(np.ceil((kw-1)/2))
        padded_images = np.pad(images,
                               pad_width=((0, 0), (p_h, p_h), (p_w, p_w)),
                               mode='constant', constant_values=0)

    convolved_images = np.zeros((m, h, w))
    for i in range(h):
        for k in range(w):
            convolved_images[:, i, k] = np.sum(padded_images[:, i:i+kh, k:k+kw]
                                               * kernel, axis=(1, 2))
    return convolved_images
