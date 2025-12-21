#!/usr/bin/env python3
"""Sparse Autoencoder Implementation with tensorflow.keras."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Create a sparse autoencoder."""
    # Encoder
    input_layer = keras.Input(shape=(input_dims,))

    x = input_layer
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)

    encoder = keras.Model(inputs=input_layer, outputs=latent,
                          name='encoder')

    # Decoder
    latent_input = keras.Input(shape=(latent_dims,))

    y = latent_input
    for units in reversed(hidden_layers):
        y = keras.layers.Dense(units, activation='relu')(y)

    output_layer = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(y)

    decoder = keras.Model(inputs=latent_input, outputs=output_layer,
                          name='decoder')

    auto_output = decoder(encoder(input_layer))
    auto = keras.Model(inputs=input_layer, outputs=auto_output,
                       name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
