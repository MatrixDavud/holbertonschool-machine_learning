#!/usr/bin/env python3
"""Building a model with keras library."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """Train a model using mini-batch gradient descent."""
    callbacks = []

    # Early stopping callback (only if enabled and validation_data exists)
    if early_stopping and validation_data is not None:
        es_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',   # monitor validation loss
            patience=patience,    # number of epochs to wait before stopping
            restore_best_weights=True  # restore model weights from best epoch
        )
        callbacks.append(es_callback)

    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch, lr):
            """Inverse time decay: lr = alpha / (1 + decay_rate * epoch)"""
            return alpha / (1 + decay_rate * epoch)
        lr_callback = K.callbacks.LearningRateScheduler(
            schedule=scheduler,
            verbose=1
        )
        callbacks.append(lr_callback)

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
