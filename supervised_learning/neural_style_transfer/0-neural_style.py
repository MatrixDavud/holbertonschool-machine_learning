#!/usr/bin/env python3
"""Neural Style Transfer Implementation."""
import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize NST class."""
        if not isinstance(style_image, np.ndarray) or \
           style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image, np.ndarray) or \
           content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescale an image so pixels are in [0, 1] and max side is 512px

        Args:
            image: numpy.ndarray with shape (h, w, 3)

        Returns:
            tf.Tensor with shape (1, h_new, w_new, 3)
        """
        if not isinstance(image, np.ndarray) or \
           image.ndim != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w = image.shape[:2]

        if h > w:
            h_new = 512
            w_new = int(w * 512 / h)
        else:
            w_new = 512
            h_new = int(h * 512 / w)

        image_resized = tf.image.resize(
            image,
            size=[h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )

        image_rescaled = image_resized / 255.0

        image_rescaled = tf.clip_by_value(image_rescaled, 0.0, 1.0)

        image_final = tf.expand_dims(image_rescaled, axis=0)

        return image_final
