#!/usr/bin/env python3
"""
0-simple_gan.py

This module defines the Simple_GAN class (a Keras Model) implementing a
basic GAN training loop where the discriminator is trained several times
per step, then the generator is trained once.

The generator tries to produce fake samples that the discriminator scores
as "real" (+1). The discriminator tries to score real samples as +1 and
fake samples as -1, using a least-squares (MSE) objective.
"""

import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """
    Simple_GAN(generator, discriminator, latent_generator, real_examples)

    A simple GAN model trained using least-squares objectives:

    - Discriminator:
        minimize MSE(D(real), +1) + MSE(D(fake), -1)

    - Generator:
        minimize MSE(D(G(z)), +1)

    Training step:
        Run `disc_iter` discriminator updates, then one generator update.
    """

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=0.005,
    ):
        """Initialize the Simple_GAN model and its optimizers."""
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # Generator objective: want D(G(z)) to be +1
        self.generator.loss = lambda x: (
            tf.keras.losses.MeanSquaredError()(
                x,
                tf.ones(x.shape),
            )
        )
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss,
        )

        # Discriminator objective: D(real)=+1, D(fake)=-1
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(
                x,
                tf.ones(x.shape),
            )
            + tf.keras.losses.MeanSquaredError()(
                y,
                -1 * tf.ones(y.shape),
            )
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss,
        )

    def get_fake_sample(self, size=None, training=False):
        """
        Generate a batch of fake samples.

        Parameters
        ----------
        size : int or None
            Number of samples to generate.
        training : bool
            Whether to run the generator in training mode.

        Returns
        -------
        tf.Tensor
            Fake samples.
        """
        if size is None:
            size = self.batch_size
        z = self.latent_generator(size)
        return self.generator(z, training=training)

    def get_real_sample(self, size=None):
        """
        Sample a batch of real examples uniformly at random.

        Parameters
        ----------
        size : int or None
            Number of real samples.

        Returns
        -------
        tf.Tensor
            Real samples.
        """
        if size is None:
            size = self.batch_size
        indices = tf.range(tf.shape(self.real_examples)[0])
        indices = tf.random.shuffle(indices)[:size]
        return tf.gather(self.real_examples, indices)

    def train_step(self, useless_argument):
        """
        Perform one GAN training step.

        The argument is unused because samples are generated internally.

        Returns
        -------
        dict
            Dictionary with discriminator and generator losses.
        """
        # Train discriminator several times
        discr_loss = None
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_batch = self.get_real_sample()
                fake_batch = self.get_fake_sample(training=False)

                d_real = self.discriminator(
                    real_batch,
                    training=True,
                )
                d_fake = self.discriminator(
                    fake_batch,
                    training=True,
                )

                discr_loss = self.discriminator.loss(
                    d_real,
                    d_fake,
                )

            grads = tape.gradient(
                discr_loss,
                self.discriminator.trainable_variables,
            )
            self.discriminator.optimizer.apply_gradients(
                zip(
                    grads,
                    self.discriminator.trainable_variables,
                )
            )

        # Train generator once
        with tf.GradientTape() as tape:
            fake_batch = self.get_fake_sample(training=True)
            d_fake = self.discriminator(
                fake_batch,
                training=False,
            )
            gen_loss = self.generator.loss(d_fake)

        grads = tape.gradient(
            gen_loss,
            self.generator.trainable_variables,
        )
        self.generator.optimizer.apply_gradients(
            zip(
                grads,
                self.generator.trainable_variables,
            )
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss,
        }
