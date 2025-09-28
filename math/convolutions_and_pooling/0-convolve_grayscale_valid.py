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
    for i in range(m):
        for k in range(conv_h*conv_w):
            row = int(k / conv_w)
            col = k % conv_w

            mat_i = images[i, row:row+kh, col:col+kw]
            res = mat_i * kernel
            s_elems = np.sum(res)

            convolved_images[i, row, col] = s_elems

    return convolved_images
