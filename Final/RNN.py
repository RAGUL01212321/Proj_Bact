import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Load data function
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove extra spaces in column names
    if 'Time' not in df.columns or 'Population' not in df.columns:
        raise KeyError(f"Required columns 'Time' and 'Population' not found in {file_path}. Available columns: {df.columns}")
    return df[['Time', 'Population']]

# Function to prepare dataset for RNN
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Set dataset path
file_path = r"C:\Users\uragu\OneDrive\Desktop\Analog Sample\datasets\nutrient_0.01.csv"

# Load the dataset
df = load_data(file_path)

# Normalize the data
scaler = MinMaxScaler()
df['Population'] = scaler.fit_transform(df[['Population']])

# Convert to numpy array
time_series = df['Population'].values

# Define sequence length
SEQ_LENGTH = 20  # Increase sequence length to capture more patterns

# Create sequences for training
X, y = create_sequences(time_series, SEQ_LENGTH)

# Reshape input for LSTM (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build RNN Model (with Dropout to prevent overfitting)
model = Sequential([
    LSTM(32, activation='tanh', return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),  # Dropout layer (20% of neurons will be randomly dropped)
    LSTM(32, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Use Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X, y, epochs=100, batch_size=16, verbose=1, callbacks=[early_stopping])

# Predict on training data
y_pred = model.predict(X)

# Inverse transform predictions
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(df['Time'][SEQ_LENGTH:], scaler.inverse_transform(y.reshape(-1, 1)), label="Actual")
plt.plot(df['Time'][SEQ_LENGTH:], y_pred, label="Predicted", linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.title("Bacterial Growth Prediction (LSTM with Dropout & Early Stopping)")
plt.show()
