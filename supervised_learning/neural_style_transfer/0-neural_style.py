#!/usr/bin/env python3
"""Neural Style Transfer Implementation."""
import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class."""

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize the NST object with style and content images."""
        if not isinstance(style_image, np.ndarray) or\
                style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a\
                             numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray) or\
                content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a\
                             numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = NST.scale_image(style_image)
        self.content_image = NST.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that the largest side is 512 pixels.

        Pixel values are in [0,1].

        Returns a tf.Tensor of shape (1, h_new, w_new, 3).
        """
        if not isinstance(image, np.ndarray) or\
                image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("image must be a\
                             numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        scale_factor = 512 / max(h, w)
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        image_resized = tf.image.resize(image, (new_h, new_w),
                                        method='bicubic')

        image_scaled = image_resized / 255.0

        return tf.expand_dims(image_scaled, axis=0)
