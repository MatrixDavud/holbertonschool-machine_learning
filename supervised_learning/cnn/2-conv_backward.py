#!/usr/bin/env python3
"""Creating Convolutional Neural Network."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Perform back propagation over a convolutional layer of a NN."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    _, h_new, w_new, _ = dZ.shape

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw),
                                 (0, 0)), mode='constant')
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            A_slice = A_prev_pad[:, vert_start:vert_end,
                                 horiz_start:horiz_end, :]

            for k in range(c_new):
                dA_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]\
                    += (
                    W[:, :, :, k] * dZ[:, i, j, k][:, None, None, None]
                )
                dW[:, :, :, k] += np.sum(
                    A_slice * dZ[:, i, j, k][:, None, None, None], axis=0
                )

    if padding == 'same':
        dA_prev = dA_prev_pad[:, ph:-ph or None, pw:-pw or None, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
