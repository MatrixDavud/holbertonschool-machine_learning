#!/usr/bin/env python3
"""Creating Convolutional Neural Network."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform back propagation over a pooling layer of a NN."""
    m, h_prev, w_prev, c = A_prev.shape
    _, h_new, w_new, _ = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            for k in range(c):
                if mode == 'max':
                    A_slice = A_prev[:, vert_start:vert_end,
                                     horiz_start:horiz_end, k]
                    mask = A_slice == np.max(A_slice, axis=(1, 2),
                                             keepdims=True)
                    dA_prev[:, vert_start:vert_end, horiz_start:horiz_end, k]\
                        += (
                        mask * dA[:, i, j, k][:, None, None]
                    )
                else:
                    da = dA[:, i, j, k]
                    average = da / (kh * kw)
                    dA_prev[:, vert_start:vert_end, horiz_start:horiz_end, k]\
                        += (
                        np.ones((m, kh, kw)) * average[:, None, None]
                    )

    return dA_prev
