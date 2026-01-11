#!/usr/bin/env python3
"""
2-wgan_gp.py

Defines the WGAN_GP class (Wasserstein GAN with Gradient Penalty).

This model improves upon WGAN_clip by enforcing the 1-Lipschitz constraint
using a gradient penalty instead of weight clipping. This results in more
stable training, especially in high-dimensional spaces.
"""

import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
    Wasserstein GAN with Gradient Penalty.

    Parameters
    ----------
    generator : tf.keras.Model
        Generator network.
    discriminator : tf.keras.Model
        Critic network (still called discriminator).
    latent_generator : callable
        Function generating latent vectors.
    real_examples : tf.Tensor
        Real dataset.
    batch_size : int
        Batch size.
    disc_iter : int
        Number of critic updates per generator update.
    learning_rate : float
        Learning rate.
    lambda_gp : float
        Gradient penalty coefficient.
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
        lambda_gp=10,
    ):
        """Initialize WGAN_GP and define losses and optimizers."""
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.3
        self.beta_2 = 0.9

        self.lambda_gp = lambda_gp

        # Dimensions handling for gradient penalty
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, dtype="int32")

        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # Generator loss (same as WGAN_clip)
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss,
        )

        # Discriminator loss (same as WGAN_clip)
        self.discriminator.loss = (
            lambda x, y: tf.reduce_mean(x) - tf.reduce_mean(y)
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
        """Generate fake samples."""
        if size is None:
            size = self.batch_size
        return self.generator(
            self.latent_generator(size),
            training=training,
        )

    def get_real_sample(self, size=None):
        """Sample real examples."""
        if size is None:
            size = self.batch_size
        indices = tf.range(tf.shape(self.real_examples)[0])
        indices = tf.random.shuffle(indices)[:size]
        return tf.gather(self.real_examples, indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """Generate interpolated samples between real and fake."""
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """Compute the gradient penalty."""
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(
                interpolated_sample,
                training=True,
            )
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(
            tf.reduce_sum(
                tf.square(grads),
                axis=self.axis,
            )
        )
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """
        Perform one WGAN-GP training step.

        Returns
        -------
        dict
            Dictionary containing discriminator loss, generator loss,
            and gradient penalty.
        """
        discr_loss = None
        gp = None

        # Train discriminator (critic)
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_batch = self.get_real_sample()
                fake_batch = self.get_fake_sample(training=False)

                interpolated = self.get_interpolated_sample(
                    real_batch,
                    fake_batch,
                )

                d_fake = self.discriminator(fake_batch, training=True)
                d_real = self.discriminator(real_batch, training=True)

                discr_loss = self.discriminator.loss(
                    d_fake,
                    d_real,
                )

                gp = self.gradient_penalty(interpolated)
                new_discr_loss = discr_loss + self.lambda_gp * gp

            grads = tape.gradient(
                new_discr_loss,
                self.discriminator.trainable_variables,
            )
            self.discriminator.optimizer.apply_gradients(
                zip(
                    grads,
                    self.discriminator.trainable_variables,
                )
            )

        # Train generator
        with tf.GradientTape() as tape:
            fake_batch = self.get_fake_sample(training=True)
            d_fake = self.discriminator(fake_batch, training=False)
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
            "gp": gp,
        }
