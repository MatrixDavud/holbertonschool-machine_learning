#!/usr/bin/env python3
"""
4-wgan_gp.py

Defines the WGAN_GP class (Wasserstein GAN with Gradient Penalty) and adds
a method to load pretrained weights from .h5 files.

The pretrained files contain:
- generator.h5: weights for the generator network
- discriminator.h5: weights for the discriminator (critic) network
"""

import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
    Wasserstein GAN with Gradient Penalty.

    This class is the same as Task 2 (WGAN_GP), with one addition:
    replace_weights(gen_h5, disc_h5) loads pretrained weights into the
    generator and discriminator.

    Parameters
    ----------
    generator : tf.keras.Model
        Generator network.
    discriminator : tf.keras.Model
        Critic network (called discriminator in this project).
    latent_generator : callable
        Function generating latent vectors of shape (k, latent_dim).
    real_examples : tf.Tensor
        Real examples tensor.
    batch_size : int
        Batch size.
    disc_iter : int
        Number of critic updates per generator update.
    learning_rate : float
        Learning rate for optimizers.
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
        """Initialize WGAN_GP, losses, and optimizers."""
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

        # Shapes used for interpolation noise u in gradient penalty.
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, dtype="int32")

        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # Generator loss: minimize -E[D(G(z))].
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

        # Critic loss: minimize E[D(fake)] - E[D(real)].
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

    def replace_weights(self, gen_h5, disc_h5):
        """
        Replace model weights using pretrained .h5 files.

        Parameters
        ----------
        gen_h5 : str
            Path to generator weights file (e.g. "generator.h5").
        disc_h5 : str
            Path to discriminator weights file (e.g. "discriminator.h5").

        Notes
        -----
        Keras requires the model variables to exist before loading weights.
        If a model was never called, variables may not be created yet.
        We ensure variables exist by doing one dummy forward pass.
        """
        # Build generator variables if needed
        try:
            _ = self.generator.trainable_variables
        except ValueError:
            _ = self.generator(self.latent_generator(1), training=False)

        # Build discriminator variables if needed
        try:
            _ = self.discriminator.trainable_variables
        except ValueError:
            dummy = self.get_fake_sample(size=1, training=False)
            _ = self.discriminator(dummy, training=False)

        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)

    def get_fake_sample(self, size=None, training=False):
        """Generate fake samples."""
        if size is None:
            size = self.batch_size
        z = self.latent_generator(size)
        return self.generator(z, training=training)

    def get_real_sample(self, size=None):
        """Sample real examples."""
        if size is None:
            size = self.batch_size
        indices = tf.range(tf.shape(self.real_examples)[0])
        indices = tf.random.shuffle(indices)[:size]
        return tf.gather(self.real_examples, indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """Create interpolated samples between real and fake."""
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """Compute gradient penalty E[(||∇D(x_hat)|| - 1)^2]."""
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
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
            {"discr_loss": ..., "gen_loss": ..., "gp": ...}
        """
        discr_loss = None
        gp = None

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

                discr_loss = self.discriminator.loss(d_fake, d_real)
                gp = self.gradient_penalty(interpolated)
                new_discr_loss = discr_loss + self.lambda_gp * gp

            grads = tape.gradient(
                new_discr_loss,
                self.discriminator.trainable_variables,
            )
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:
            fake_batch = self.get_fake_sample(training=True)
            d_fake = self.discriminator(fake_batch, training=False)
            gen_loss = self.generator.loss(d_fake)

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
