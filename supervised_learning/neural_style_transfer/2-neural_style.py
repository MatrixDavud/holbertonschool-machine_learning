#!/usr/bin/env python3
"""Neural Style Transfer Implementation."""
import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize class object."""
        valid = (isinstance(style_image, np.ndarray)
                 and style_image.ndim == 3 and style_image.shape[2] == 3)
        if not valid:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        valid_img = (isinstance(content_image, np.ndarray)
                     and content_image.ndim == 3
                     and content_image.shape[2] == 3)
        if not valid_img:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescale an image so pixels are in [0, 1] and max side is 512px.

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

    def load_model(self):
        """Create the model used to calculate cost."""
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet',
        )
        vgg.trainable = False

        for layer in vgg.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer.__class__ = tf.keras.layers.AveragePooling2D

        style_outputs = [
            vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        self.model = tf.keras.Model(
            inputs=vgg.input,
            outputs=[content_output] + style_outputs
        )

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the Gram matrix of a given layer output."""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        input_layer = tf.reshape(input_layer, [-1, input_layer.shape[-1]])
        gram = tf.matmul(input_layer, input_layer, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        return gram / tf.cast(tf.shape(input_layer)[0], tf.float32)
