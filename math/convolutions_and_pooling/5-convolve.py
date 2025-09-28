#!/usr/bin/env python3
"""Convolution on images with multiple kernels."""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Perform a convolution on images using multiple kernels."""
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if c != kc:
        raise ValueError("Channel number in kernel must match image channel.")

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + (((h - 1) * sh + kh - h) % 2)
        pw = ((w - 1) * sw + kw - w) // 2 + (((w - 1) * sw + kw - w) % 2)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_images = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    h_padded, w_padded = padded_images.shape[1], padded_images.shape[2]
    conv_h = (h_padded - kh) // sh + 1
    conv_w = (w_padded - kw) // sw + 1

    convolved_images = np.zeros((m, conv_h, conv_w, nc))

    for i in range(conv_h):
        for j in range(conv_w):
            for k in range(nc):
                region = padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                convolved_images[:, i, j, k] = \
                    np.sum(region * kernels[:, :, :, k], axis=(1, 2, 3))

    return convolved_images
