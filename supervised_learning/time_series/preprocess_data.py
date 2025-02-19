#!/usr/bin/env python3
"""
Preprocess Bitcoin price data for time series forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Loads BTC dataset from a CSV file."""
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    return df

def preprocess_data(df):
    """Prepares the BTC dataset for forecasting."""
    df = df[['close']]
    df.dropna(inplace=True)
    scaler = MinMaxScaler()
    df['close'] = scaler.fit_transform(df[['close']])
    return df, scaler

def create_sequences(data, seq_length=24):
    """Creates time series sequences from the dataset."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    coinbase = load_data("coinbase.csv")
    bitstamp = load_data("bitstamp.csv")
    df = pd.concat([coinbase, bitstamp])
    df, scaler = preprocess_data(df)
    X, y = create_sequences(df['close'].values)
    np.save("X.npy", X)
    np.save("y.npy", y)
    np.save("scaler.npy", scaler.scale_)
