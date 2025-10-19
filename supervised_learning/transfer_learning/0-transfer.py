#!/usr/bin/env python3
"""
Transfer learning script for CIFAR-10 classification using Keras Applications.
Uses EfficientNetB0 with data augmentation and fine-tuning to achieve >87% validation accuracy.
"""

import numpy as np
from tensorflow import keras as K
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model.
    
    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data
        Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    
    Returns:
        X_p: numpy.ndarray containing the preprocessed X
        Y_p: numpy.ndarray containing the preprocessed Y
    """
    # Preprocess X using EfficientNet's preprocessing function
    X_p = preprocess_input(X.astype('float32'))
    
    # Convert labels to categorical one-hot encoding
    Y_p = K.utils.to_categorical(Y, 10)
    
    return X_p, Y_p


def create_model(input_shape=(32, 32, 3), dense_units=256):
    """
    Creates a model using EfficientNetB0 as base with transfer learning.
    
    Args:
        input_shape: Shape of input images
        dense_units: Number of units in the dense layer
    
    Returns:
        Compiled Keras model and base model
    """
    # Input layer
    inputs = K.Input(shape=input_shape)
    
    # Lambda layer to resize images from 32x32 to 224x224 (EfficientNetB0 input size)
    x = layers.Lambda(
        lambda image: tf.image.resize(image, (224, 224)),
        output_shape=(224, 224, 3),
        input_shape=(32, 32, 3)
    )(inputs)
    
    # Load EfficientNetB0 without top layers, pre-trained on ImageNet
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=x,
        pooling='avg'
    )
    
    # Freeze the base model layers initially
    base_model.trainable = False
    
    # Add custom classification layers with more capacity
    x = base_model.output
    x = layers.Dense(dense_units, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
    
    # Create the model
    model = K.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model


def create_data_augmentation():
    """
    Creates a data augmentation pipeline.
    
    Returns:
        Sequential model with augmentation layers
    """
    return K.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name='data_augmentation')


if __name__ == '__main__':
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    
    # Preprocess the data
    print("Preprocessing data...")
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)
    
    # Create the model with larger dense layer
    print("Creating model with EfficientNetB0...")
    model, base_model = create_model(dense_units=256)
    
    print("\nModel architecture:")
    model.summary()
    
    # ==============================================================================
    # PHASE 1: Train top layers with frozen base using feature extraction
    # ==============================================================================
    print("\n" + "="*80)
    print("PHASE 1: Training top layers with frozen base (feature extraction)")
    print("="*80)
    
    # Create a feature extraction model (just the frozen base)
    feature_extractor = K.Model(
        inputs=model.input,
        outputs=base_model.output
    )
    
    # Extract features once (Hint 3: compute frozen layers output once)
    print("\nExtracting training features...")
    train_features = feature_extractor.predict(X_train_p, batch_size=128, verbose=1)
    print("Extracting validation features...")
    test_features = feature_extractor.predict(X_test_p, batch_size=128, verbose=1)
    
    # Create a new model that trains only the top layers with matching layer names
    print("\nCreating model for training top layers only...")
    feature_input = K.Input(shape=train_features.shape[1:])
    x = layers.Dense(256, activation='relu', name='fc1')(feature_input)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
    
    top_model = K.Model(inputs=feature_input, outputs=outputs)
    top_model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train only the top layers on extracted features
    print("\nTraining top layers (Phase 1)...")
    callbacks_phase1 = [
        K.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history1 = top_model.fit(
        train_features,
        Y_train_p,
        batch_size=128,
        epochs=50,
        validation_data=(test_features, Y_test_p),
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # Transfer the trained weights back to the full model
    print("\nTransferring weights to full model...")
    for layer_name in ['fc1', 'bn1', 'fc2', 'predictions']:
        weights = top_model.get_layer(layer_name).get_weights()
        model.get_layer(layer_name).set_weights(weights)
    
    # Evaluate after Phase 1
    print("\nEvaluating after Phase 1...")
    test_loss, test_acc = model.evaluate(X_test_p, Y_test_p, batch_size=128, verbose=1)
    print(f"Phase 1 Test accuracy: {test_acc:.4f}")
    
    # ==============================================================================
    # PHASE 2: Fine-tune with data augmentation
    # ==============================================================================
    print("\n" + "="*80)
    print("PHASE 2: Fine-tuning with data augmentation")
    print("="*80)
    
    # Unfreeze the last few layers of the base model for fine-tuning
    base_model.trainable = True
    
    # Freeze all layers except the last 20
    print(f"\nTotal base model layers: {len(base_model.layers)}")
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Unfrozen base model layers: {trainable_count}")
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel summary after unfreezing:")
    print(f"Total params: {model.count_params():,}")
    print(f"Trainable params: {sum([K.backend.count_params(w) for w in model.trainable_weights]):,}")
    print(f"Non-trainable params: {sum([K.backend.count_params(w) for w in model.non_trainable_weights]):,}")
    
    # Create data augmentation
    data_aug = create_data_augmentation()
    
    # Create augmented dataset
    print("\nApplying data augmentation...")
    
    # Manual data augmentation during training
    def augment_data(X, Y, augmentation_model, augment_ratio=0.5):
        """Augment a portion of the training data"""
        n_samples = X.shape[0]
        n_augment = int(n_samples * augment_ratio)
        
        # Select random samples to augment
        indices = np.random.choice(n_samples, n_augment, replace=False)
        X_aug = X[indices]
        Y_aug = Y[indices]
        
        # Apply augmentation
        X_aug = augmentation_model(X_aug, training=True).numpy()
        
        # Combine original and augmented data
        X_combined = np.concatenate([X, X_aug], axis=0)
        Y_combined = np.concatenate([Y, Y_aug], axis=0)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X_combined))
        return X_combined[shuffle_idx], Y_combined[shuffle_idx]
    
    # Augment training data
    X_train_aug, Y_train_aug = augment_data(X_train_p, Y_train_p, data_aug)
    print(f"Augmented training set size: {X_train_aug.shape[0]}")
    
    # Fine-tune with augmented data
    callbacks_phase2 = [
        K.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1
        )
    ]
    
    print("\nFine-tuning model (Phase 2)...")
    history2 = model.fit(
        X_train_aug,
        Y_train_aug,
        batch_size=64,
        epochs=30,
        validation_data=(X_test_p, Y_test_p),
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    # ==============================================================================
    # Final Evaluation and Saving
    # ==============================================================================
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    # Final evaluation
    print("\nEvaluating final model on test set...")
    test_loss, test_acc = model.evaluate(X_test_p, Y_test_p, batch_size=128, verbose=1)
    print(f"\nFinal Test accuracy: {test_acc:.4f}")
    
    # Save the model
    print("\nSaving model to cifar10.h5...")
    model.save('cifar10.h5')
    print("Model saved successfully!")
    
    # Verify the saved model
    print("\nVerifying saved model...")
    loaded_model = K.models.load_model('cifar10.h5')
    verify_loss, verify_acc = loaded_model.evaluate(X_test_p, Y_test_p, batch_size=128, verbose=0)
    print(f"Loaded model test accuracy: {verify_acc:.4f}")
    
    if verify_acc >= 0.87:
        print("\n" + "="*80)
        print("✓ SUCCESS: Model achieves >87% validation accuracy!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print(f"✗ Model accuracy {verify_acc:.4f} is below 87% target")
        print("Consider training for more epochs or adjusting hyperparameters")
        print("="*80)