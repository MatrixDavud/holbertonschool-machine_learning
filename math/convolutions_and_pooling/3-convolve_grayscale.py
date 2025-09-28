#!/usr/bin/env python3
"""General convolution for grayscale images."""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Perform a convolution on grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + (((h - 1) * sh + kh - h) % 2)
        pw = ((w - 1) * sw + kw - w) // 2 + (((w - 1) * sw + kw - w) % 2)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_images = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    h_padded, w_padded = padded_images.shape[1], padded_images.shape[2]
    conv_h = (h_padded - kh) // sh + 1
    conv_w = (w_padded - kw) // sw + 1

    convolved_images = np.zeros((m, conv_h, conv_w))

    for i in range(conv_h):
        for j in range(conv_w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw] * kernel,
                axis=(1, 2)
            )

    return convolved_images
