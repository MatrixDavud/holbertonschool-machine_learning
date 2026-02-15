#!/usr/bin/env python3
"""
4`creates masks
"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    creates all masks for training/validation
    """
    # Encoder padding mask
    encoder_padding_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_padding_mask[:, tf.newaxis, tf.newaxis, :]
    # Decoder padding mask
    decoder_padding_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_padding_mask[:, tf.newaxis, tf.newaxis, :]
    # combined mask
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((target.shape[1], target.shape[1])), -1, 0)
    dec_target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    combined_mask = tf.maximum(
        dec_target_padding_mask[:, tf.newaxis, tf.newaxis, :], look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
