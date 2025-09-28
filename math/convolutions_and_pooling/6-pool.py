#!/usr/bin/env python3
"""Convolution on images with multiple kernels."""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Perform pooling on images."""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    pooled = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            window = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(window, axis=(1, 2))

    return pooled
