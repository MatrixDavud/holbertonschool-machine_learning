#!/usr/bin/env python3
"""
1-wgan_clip.py

Defines the WGAN_clip class (Wasserstein GAN with weight clipping).

Key differences from a "simple" GAN:
- The discriminator becomes a *critic* (no probability interpretation).
- Losses are based on the Wasserstein-1 (Earth Mover) distance estimate.
- The critic weights are clipped in [-1, 1] to enforce a Lipschitz constraint.

Training step:
- Update critic `disc_iter` times
- Clip critic weights after each critic update
- Update generator once
"""

import tensorflow as tf
from tensorflow import keras


class WGAN_clip(keras.Model):
    """
    Wasserstein GAN with weight clipping.

    Parameters
    ----------
    generator : tf.keras.Model
        Generator network G(z).
    discriminator : tf.keras.Model
        Critic network D(x). (Still named discriminator in the project.)
    latent_generator : callable
        Function k -> latent batch of shape (k, latent_dim).
    real_examples : tf.Tensor
        Real samples tensor of shape (N, data_dim).
    batch_size : int
        Batch size.
    disc_iter : int
        Number of critic updates per train step.
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
        """Initialize WGAN_clip and define losses/optimizers."""
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

        # WGAN generator loss:
        # minimize -E[D(G(z))]  <=> maximize E[D(G(z))]
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss,
        )

        # WGAN critic (discriminator) loss:
        # minimize E[D(fake)] - E[D(real)]
        # (equivalent to maximizing E[D(real)] - E[D(fake)])
        self.discriminator.loss = (
            lambda x, y: tf.math.reduce_mean(x) - tf.math.reduce_mean(y)
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
        Generate fake samples using the generator.

        Parameters
        ----------
        size : int or None
            Number of samples. Defaults to self.batch_size.
        training : bool
            Whether to run generator in training mode.

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
        Sample real examples uniformly at random.

        Parameters
        ----------
        size : int or None
            Number of samples. Defaults to self.batch_size.

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

    def _clip_discriminator_weights(self):
        """
        Clip critic parameters into [-1, 1].

        WGAN-clip enforces the 1-Lipschitz constraint approximately by
        clipping weights after each critic update.
        """
        for var in self.discriminator.trainable_variables:
            var.assign(tf.clip_by_value(var, -1.0, 1.0))

    def train_step(self, useless_argument):
        """
        Perform one WGAN training step.

        Returns
        -------
        dict
            {"discr_loss": ..., "gen_loss": ...}
        """
        discr_loss = None

        # 1) Train critic multiple times
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_batch = self.get_real_sample()
                fake_batch = self.get_fake_sample(training=False)

                d_fake = self.discriminator(fake_batch, training=True)
                d_real = self.discriminator(real_batch, training=True)

                # loss = E[D(fake)] - E[D(real)]
                discr_loss = self.discriminator.loss(d_fake, d_real)

            grads = tape.gradient(
                discr_loss,
                self.discriminator.trainable_variables,
            )
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

            # New part vs Simple_GAN: weight clipping
            self._clip_discriminator_weights()

        # 2) Train generator once
        with tf.GradientTape() as tape:
            fake_batch = self.get_fake_sample(training=True)
            d_fake = self.discriminator(fake_batch, training=False)
            gen_loss = self.generator.loss(d_fake)

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
