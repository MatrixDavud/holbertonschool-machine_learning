#!/usr/bin/env python3
"""Variational Autoencoder Implementation with tensorflow.keras."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder."""
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units=units, activation="relu")(x)

    # Latent space
    z_mean = keras.layers.Dense(units=latent_dims, name="z_mean")(x)
    z_log_var = keras.layers.Dense(units=latent_dims, name="z_log_var")(x)

    def sampling(args):
        """Samples from the latent space using the reparameterization."""
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(z_mean))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(
        sampling, output_shape=(latent_dims,), name="z")([z_mean, z_log_var])
    encoder = keras.Model(
        inputs=encoder_inputs, outputs=[z, z_mean, z_log_var], name="encoder")

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units=units, activation="relu")(x)

    decoder_outputs = keras.layers.Dense(
        units=input_dims, activation="sigmoid")(x)
    decoder = keras.Model(
        inputs=decoder_inputs, outputs=decoder_outputs, name="decoder")

    # VAE Model
    z, z_mean, z_log_var = encoder(encoder_inputs)
    reconstructed = decoder(z)
    auto = keras.Model(
        inputs=encoder_inputs, outputs=reconstructed, name="vae")

    # Loss Function
    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_inputs, reconstructed)
    reconstruction_loss *= input_dims
    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_var - keras.backend.square(
            z_mean) - keras.backend.exp(z_log_var), axis=-1)
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    # Compile Model
    auto.compile(optimizer=keras.optimizers.Adam())

    return encoder, decoder, auto
