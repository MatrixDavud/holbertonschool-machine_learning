#!/usr/bin/env python3
"""Convolutional Autoencoder Implementation with tensorflow.keras."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Create a convolutional autoencoder."""
    # Encoder
    input_layer = keras.Input(shape=input_dims)
    x = input_layer

    for f in filters:
        x = keras.layers.Conv2D(
            f,
            (3, 3),
            activation='relu',
            padding='same'
        )(x)
        x = keras.layers.MaxPooling2D(
            (2, 2),
            padding='same'
        )(x)

    latent = x

    encoder = keras.Model(
        inputs=input_layer,
        outputs=latent,
        name='encoder'
    )

    # Decoder
    latent_input = keras.Input(shape=latent_dims)
    y = latent_input

    rev_filters = filters[::-1]

    for i, f in enumerate(rev_filters):
        if i == len(rev_filters) - 1:
            y = keras.layers.Conv2D(
                f,
                (3, 3),
                activation='relu',
                padding='valid'
            )(y)
            y = keras.layers.UpSampling2D(
                (2, 2)
            )(y)
        else:
            y = keras.layers.Conv2D(
                f,
                (3, 3),
                activation='relu',
                padding='same'
            )(y)
            y = keras.layers.UpSampling2D(
                (2, 2)
            )(y)

    output_layer = keras.layers.Conv2D(
        input_dims[-1],
        (3, 3),
        activation='sigmoid',
        padding='same'
    )(y)

    decoder = keras.Model(
        inputs=latent_input,
        outputs=output_layer,
        name='decoder'
    )

    auto_output = decoder(encoder(input_layer))

    auto = keras.Model(
        inputs=input_layer,
        outputs=auto_output,
        name='autoencoder'
    )

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
