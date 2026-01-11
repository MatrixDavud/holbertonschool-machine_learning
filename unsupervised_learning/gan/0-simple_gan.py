#!/usr/bin/env python3
"""
0-simple_gan.py

This module defines the Simple_GAN class (a Keras Model) implementing a basic
GAN training loop where the discriminator is trained several times per step,
then the generator is trained once.

The generator tries to produce fake samples that the discriminator scores as
"real" (+1). The discriminator tries to score real samples as +1 and fake
samples as -1, using a least-squares (MSE) objective.
"""

import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """
    Simple_GAN(generator, discriminator, latent_generator, real_examples, ...)

    A simple GAN model trained using the "least squares" objectives:

    - Discriminator:
        minimize MSE(D(real), +1) + MSE(D(fake), -1)

    - Generator:
        minimize MSE(D(G(z)), +1)

    Training step:
      Run `disc_iter` discriminator updates, then 1 generator update.

    Parameters
    ----------
    generator : tf.keras.Model
        The generator network G(z).
    discriminator : tf.keras.Model
        The discriminator network D(x).
    latent_generator : callable
        A function that takes an integer k and returns a tensor of latent vectors
        of shape (k, latent_dim).
    real_examples : tf.Tensor
        A dataset tensor containing real samples, shape (N, data_dim).
    batch_size : int
        Batch size for real/fake sampling.
    disc_iter : int
        Number of discriminator updates per train step.
    learning_rate : float
        Learning rate for Adam optimizers.
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
        """Initialize the Simple_GAN model and define losses/optimizers."""
        super().__init__()  # initializes keras.Model internals (e.g., history)
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
        self.generator.loss = (
            lambda x: tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
        )
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2
        )
        self.generator.compile(optimizer=self.generator.optimizer, loss=self.generator.loss)

        # Discriminator objective: want D(real)=+1 and D(fake)=-1
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
            + tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape))
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer, loss=self.discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """
        Generate a batch of fake samples.

        Parameters
        ----------
        size : int or None
            Number of samples to generate. If None, uses self.batch_size.
        training : bool
            Passed to the generator (affects layers like Dropout/BatchNorm).

        Returns
        -------
        tf.Tensor
            Fake samples of shape (size, data_dim).
        """
        if not size:
            size = self.batch_size
        z = self.latent_generator(size)
        return self.generator(z, training=training)

    def get_real_sample(self, size=None):
        """
        Sample a batch of real examples uniformly at random from self.real_examples.

        Parameters
        ----------
        size : int or None
            Number of real samples to draw. If None, uses self.batch_size.

        Returns
        -------
        tf.Tensor
            Real samples of shape (size, data_dim).
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Perform one GAN training step.

        Keras calls this method repeatedly inside model.fit(). The argument is
        unused here because we sample real data from self.real_examples and
        fake data from the latent generator.

        Returns
        -------
        dict
            Dictionary with keys "discr_loss" and "gen_loss" for Keras logging.
        """
        # 1) Update discriminator disc_iter times
        discr_loss = None
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # Real batch
                real_batch = self.get_real_sample(training=False) if False else self.get_real_sample()
                # Fake batch (do NOT train generator while training discriminator)
                fake_batch = self.get_fake_sample(training=False)

                # Discriminator outputs
                d_real = self.discriminator(real_batch, training=True)
                d_fake = self.discriminator(fake_batch, training=True)

                # Discriminator loss: want d_real -> +1, d_fake -> -1
                discr_loss = self.discriminator.loss(d_real, d_fake)

            grads = tape.gradient(discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        # 2) Update generator once
        with tf.GradientTape() as tape:
            # Fake batch, generator in training mode (so its params get gradients)
            fake_batch = self.get_fake_sample(training=True)

            # Discriminator score of generated samples
            d_fake = self.discriminator(fake_batch, training=False)

            # Generator loss: want discriminator to output +1 on fake samples
            gen_loss = self.generator.loss(d_fake)

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
