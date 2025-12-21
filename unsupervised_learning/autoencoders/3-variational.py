#!/usr/bin/env python3
"""Variational Autoencoder Implementation with tensorflow.keras."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder."""
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Sampling function for the reparameterization trick
    def sampling(args):
        mu, log_var = args
        batch = keras.backend.shape(mu)[0]
        dim = keras.backend.int_shape(mu)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return mu + keras.backend.exp(0.5 * log_var) * epsilon

    # Latent space output
    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,))([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs,
                          [z, z_mean, z_log_var], name='encoder')

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')

    # --- Full Autoencoder ---
    auto_outputs = decoder(encoder(encoder_inputs)[0])
    auto = keras.Model(encoder_inputs, auto_outputs, name='vae')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
