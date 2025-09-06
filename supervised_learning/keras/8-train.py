#!/usr/bin/env python3
"""Building a model with keras library."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """Train a model using mini-batch gradient descent."""
    callbacks = []

    # Early stopping callback
    if early_stopping and validation_data is not None:
        es_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(es_callback)

    # Learning rate decay callback
    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch, lr):
            """Inverse time decay: lr = alpha / (1 + decay_rate * epoch)"""
            return alpha / (1 + decay_rate * epoch)

        lr_callback = K.callbacks.LearningRateScheduler(
            schedule=scheduler,
            verbose=1
        )
        callbacks.append(lr_callback)

    # Model checkpoint callback (save best model)
    if save_best and validation_data is not None and filepath is not None:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            save_weights_only=False,
            verbose=0
        )
        callbacks.append(checkpoint)

    # Train the model
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history
