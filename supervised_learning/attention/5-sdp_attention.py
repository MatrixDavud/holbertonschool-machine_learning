#!/usr/bin/env python3
"""
Scaled dot product attention
"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention

    Args:
        Q (tf.Tensor): query tensor (..., seq_len_q, dk)
        K (tf.Tensor): key tensor (..., seq_len_v, dk)
        V (tf.Tensor): value tensor (..., seq_len_v, dv)
        mask (tf.Tensor, optional): mask tensor broadcastable to
                                    (..., seq_len_q, seq_len_v)

    Returns:
        output (tf.Tensor): attention output (..., seq_len_q, dv)
        weights (tf.Tensor): attention weights (..., seq_len_q, seq_len_v)
    """
    # Matrix multiplication of Q and K^T
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale by sqrt(dk)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply mask if provided
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax to get attention weights
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Weighted sum of values
    output = tf.matmul(weights, V)

    return output, weights
