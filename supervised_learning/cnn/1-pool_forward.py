#!/usr/bin/env python3
"""Creating Convolutional Neural Network."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Propogate forward over a pooling layer of a NN."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = int((h_prev - kh) / sh) + 1
    w_new = int((w_prev - kw) / sw) + 1

    A = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw
            A_slice = A_prev[:, vert_start:vert_end, horiz_start:horiz_end, :]
            if mode == 'max':
                A[:, i, j, :] = np.max(A_slice, axis=(1, 2))
            else:
                A[:, i, j, :] = np.mean(A_slice, axis=(1, 2))

    return A
