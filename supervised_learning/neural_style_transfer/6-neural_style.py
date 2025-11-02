#!/usr/bin/env python3
"""
6-neural_style.py
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    class to perform neural style transfer
    """
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        class constructor
        """
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
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        scales an image such that its
        """
        valid = (isinstance(image, np.ndarray)
                 and image.ndim == 3 and image.shape[2] == 3)
        if not valid:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        max_dim = max(h, w)
        scale = 512 / max_dim
        new_size = (int(h * scale), int(w * scale))
        # Convert to tensor and add batch dimension
        image_tensor = tf.convert_to_tensor(image)[tf.newaxis, ...]

        # Resize using bicubic interpolation (while in [0, 255] range)
        resized_image = tf.image.resize(
            image_tensor,
            new_size,
            method=tf.image.ResizeMethod.BICUBIC
        )

        # rescale to [0, 1]
        scaled_image = tf.clip_by_value(resized_image / 255.0, 0.0, 1.0)

        return scaled_image

    def load_model(self):
        """
        creates the model used to calculate cost
        """
        # Load VGG19 without top layers and pretrained on ImageNet
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        # Replace MaxPooling2D layers with AveragePooling2D
        pooling_layers = {"MaxPooling2D": tf.keras.layers.AveragePooling2D}
        vgg.save("base_vgg")
        # Reload the VGG model with the pooling layers swapped
        vgg = tf.keras.models.load_model("base_vgg",
                                         custom_objects=pooling_layers)
        # Make sure that the model is non-trainable
        vgg.trainable = False
        # Get the desired layer outputs
        outputs = [vgg.get_layer(layer).output for layer in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)
        # Create a model that returns the outputs
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        calculate gram matrices
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        # Unroll the input layer
        input_layer = tf.reshape(input_layer, [-1, input_layer.shape[-1]])
        # Calculate the Gram matrix
        gram = tf.matmul(input_layer, input_layer, transpose_a=True)
        # Normalize by the number of elements in the batch
        gram = tf.expand_dims(gram, axis=0)  # Add batch dimension
        return gram / tf.cast(tf.shape(input_layer)[0], tf.float32)

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost
        """
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        # Get the style features and content features
        # For style image outputs
        style_outputs = self.model(style_image)
        self.gram_style_features = [self.gram_matrix(style_feature)
                                    for style_feature in style_outputs[:-1]]

        # For content image outputs
        self.content_feature = self.model(content_image)[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError("style_output must be a tensor of rank 4")
        if len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")
        ch = style_output.shape[-1]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {ch}, {ch}]")
        if len(gram_target.shape) != 3 or gram_target.shape != [1, ch, ch]:
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {ch}, {ch}]")

        # Calculate the Gram matrix for the style output
        gram_style = self.gram_matrix(style_output)
        # Calculate the style cost
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for all layers
        """
        L = len(self.style_layers)
        if not isinstance(style_outputs, list):
            raise TypeError(
                f"style_outputs must be a list with a length of {L}")
        if len(style_outputs) != L:
            raise TypeError(
                f"style_outputs must be a list with a length of {L}")
            raise TypeError(
                f"style_outputs must be a list with a length of {L}")
        # Calculate the style cost for each layer
        style_cost = 0
        for target, output in zip(self.gram_style_features, style_outputs):
            style_cost += self.layer_style_cost(output, target)
        return style_cost / L

    def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image
        """
        s = self.content_feature.shape
        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError(f"content_output must be a tensor of shape {s}")
        if len(content_output.shape) != len(self.content_feature.shape):
            raise TypeError(f"content_output must be a tensor of shape {s}")
        # Calculate the content cost
        return tf.reduce_mean(tf.square(content_output - self.content_feature))
