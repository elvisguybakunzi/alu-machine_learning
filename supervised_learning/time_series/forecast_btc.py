#!/usr/bin/env python3
"""
Forecast Bitcoin price using an RNN model.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load preprocessed data
X = np.load("X.npy")
y = np.load("y.npy")
scaler_scale = np.load("scaler.npy")

# Reshape input for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build RNN model
def build_model(input_shape):
    """Creates and compiles the LSTM model."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train the model
model = build_model((X.shape[1], X.shape[2]))
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model.save("btc_forecast_model.h5")
