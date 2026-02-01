#!/usr/bin/env python3
"""
Preprocess Bitcoin data for time series forecasting
"""
import pandas as pd
import numpy as np


def preprocess_data():
    """
    Preprocesses Bitcoin data from coinbase and bitstamp datasets
    
    The preprocessing includes:
    - Loading raw data
    - Removing rows with NaN values
    - Selecting useful features (Close price primarily, with volume data)
    - Interpolating missing values if needed
    - Normalizing the data
    - Saving preprocessed data
    
    Returns:
        None (saves preprocessed data to file)
    """
    # Load the datasets
    print("Loading datasets...")
    try:
        coinbase = pd.read_csv('coinbase.csv')
    except FileNotFoundError:
        print("Error: coinbase.csv not found")
        coinbase = None
    
    try:
        bitstamp = pd.read_csv('bitstamp.csv')
    except FileNotFoundError:
        print("Error: bitstamp.csv not found")
        bitstamp = None
    
    if coinbase is None and bitstamp is None:
        raise FileNotFoundError("No data files found")
    
    # Combine datasets if both exist
    if coinbase is not None and bitstamp is not None:
        print("Combining coinbase and bitstamp data...")
        # Align on timestamp and average the values
        coinbase_clean = coinbase.dropna()
        bitstamp_clean = bitstamp.dropna()
        
        # Merge on timestamp
        combined = pd.merge(
            coinbase_clean, 
            bitstamp_clean, 
            on='Timestamp', 
            how='outer',
            suffixes=('_coinbase', '_bitstamp')
        )
        combined = combined.sort_values('Timestamp').reset_index(drop=True)
        
        # Average the price columns
        combined['Close'] = combined[['Close_coinbase', 'Close_bitstamp']].mean(axis=1)
        combined['High'] = combined[['High_coinbase', 'High_bitstamp']].mean(axis=1)
        combined['Low'] = combined[['Low_coinbase', 'Low_bitstamp']].mean(axis=1)
        combined['Open'] = combined[['Open_coinbase', 'Open_bitstamp']].mean(axis=1)
        combined['Volume_(BTC)'] = combined[['Volume_(BTC)_coinbase', 
                                              'Volume_(BTC)_bitstamp']].sum(axis=1)
        
        # Keep only necessary columns
        data = combined[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)']].copy()
    elif coinbase is not None:
        print("Using coinbase data only...")
        data = coinbase[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)']].copy()
    else:
        print("Using bitstamp data only...")
        data = bitstamp[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)']].copy()
    
    # Remove rows with NaN values
    print(f"Data shape before removing NaN: {data.shape}")
    data = data.dropna()
    print(f"Data shape after removing NaN: {data.shape}")
    
    # Sort by timestamp to ensure chronological order
    data = data.sort_values('Timestamp').reset_index(drop=True)
    
    # Forward fill any remaining gaps (if any isolated NaNs remain)
    data = data.fillna(method='ffill')
    
    # Remove duplicates based on timestamp
    data = data.drop_duplicates(subset=['Timestamp'], keep='first')
    
    # Feature engineering - add useful derived features
    # We'll keep the original features and let the model learn from them
    # The most important feature is Close price, but High, Low, Open contain info
    
    # Select features for modeling
    # We'll use: Open, High, Low, Close, Volume_(BTC)
    features = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)']
    
    # Store min and max for denormalization later
    stats = {}
    for feature in features:
        stats[f'{feature}_min'] = data[feature].min()
        stats[f'{feature}_max'] = data[feature].max()
    
    # Min-Max normalization to [0, 1]
    print("Normalizing data...")
    for feature in features:
        min_val = data[feature].min()
        max_val = data[feature].max()
        data[feature] = (data[feature] - min_val) / (max_val - min_val)
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    data.to_csv('preprocessed_data.csv', index=False)
    
    # Save statistics for denormalization
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv('normalization_stats.csv', index=False)
    
    print(f"Preprocessing complete. Final data shape: {data.shape}")
    print(f"Time range: {data['Timestamp'].min()} to {data['Timestamp'].max()}")
    
    return data


if __name__ == "__main__":
    preprocess_data()
