#!/usr/bin/env python3
"""Performing convolutions on images."""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Perform a convolution on grayscale images with custom padding."""
    kh, kw = kernel.shape
    ph = padding[0]
    pw = padding[1]
    padded_images = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)
    m, h, w = padded_images.shape
    conv_h = h-kh+1
    conv_w = w-kw+1

    conv_images = np.zeros((m, conv_h, conv_w))
    for i in range(conv_h):
        for k in range(conv_w):
            conv_images[:, i, k] = np.sum(padded_images[:, i:i+kh, k:k+kw]
                                          * kernel, axis=(1, 2))
    return conv_images
