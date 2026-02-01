#!/usr/bin/env python3
"""
Bitcoin price forecasting using RNN
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def create_sequences(data, lookback=24*60):
    """
    Create sequences for time series forecasting
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        lookback: number of timesteps to look back (24 hours * 60 minutes)
    
    Returns:
        X: input sequences of shape (n_samples, lookback, n_features)
        y: target values (Close price at next timestep)
    """
    X, y = [], []
    
    for i in range(lookback, len(data)):
        # Use past 'lookback' timesteps as input
        X.append(data[i-lookback:i])
        # Predict the Close price at the next timestep
        # Close price is at index 3 (Open, High, Low, Close, Volume)
        y.append(data[i, 3])
    
    return np.array(X), np.array(y)


def build_model(input_shape):
    """
    Build LSTM model for Bitcoin forecasting
    
    Args:
        input_shape: tuple of (timesteps, features)
    
    Returns:
        compiled keras model
    """
    model = keras.Sequential([
        # First LSTM layer with return sequences
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        
        # Second LSTM layer
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.1),
        
        # Output layer (single value - predicted Close price)
        layers.Dense(1)
    ])
    
    # Compile with MSE loss as specified
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def prepare_dataset(X, y, batch_size=32, validation_split=0.2):
    """
    Create tf.data.Dataset for training
    
    Args:
        X: input sequences
        y: target values
        batch_size: batch size for training
        validation_split: fraction of data to use for validation
    
    Returns:
        train_dataset, val_dataset
    """
    # Split into train and validation
    split_idx = int(len(X) * (1 - validation_split))
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset


def main():
    """
    Main function to preprocess, train, and validate the model
    """
    print("Loading preprocessed data...")
    try:
        data = pd.read_csv('preprocessed_data.csv')
    except FileNotFoundError:
        print("Error: preprocessed_data.csv not found. Please run preprocess_data.py first.")
        return
    
    # Select features (exclude Timestamp)
    features = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)']
    data_values = data[features].values
    
    print(f"Data shape: {data_values.shape}")
    
    # Create sequences (24 hours = 24 * 60 minutes = 1440 timesteps)
    lookback = 24 * 60  # 24 hours of 60-second windows
    print(f"Creating sequences with lookback of {lookback} timesteps (24 hours)...")
    
    X, y = create_sequences(data_values, lookback=lookback)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    if len(X) == 0:
        print("Not enough data to create sequences. Need at least 24 hours of continuous data.")
        return
    
    # Prepare datasets
    batch_size = 32
    train_dataset, val_dataset = prepare_dataset(X, y, batch_size=batch_size)
    
    # Build model
    print("Building model...")
    model = build_model(input_shape=(lookback, X.shape[2]))
    model.summary()
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save model
    print("Saving model...")
    model.save('btc_forecast_model.h5')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Model Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Model MAE')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to training_history.png")
    
    # Evaluate on validation set
    print("\nEvaluating model on validation set...")
    val_loss, val_mae = model.evaluate(val_dataset)
    print(f"Validation MSE: {val_loss:.6f}")
    print(f"Validation MAE: {val_mae:.6f}")
    
    # Load normalization stats to denormalize predictions
    try:
        stats = pd.read_csv('normalization_stats.csv')
        close_min = stats['Close_min'].values[0]
        close_max = stats['Close_max'].values[0]
        
        # Denormalize MAE for interpretability
        denorm_mae = val_mae * (close_max - close_min)
        print(f"Validation MAE (denormalized): ${denorm_mae:.2f}")
    except:
        print("Could not load normalization stats for denormalization")
    
    print("\nModel training complete!")


if __name__ == "__main__":
    main()
