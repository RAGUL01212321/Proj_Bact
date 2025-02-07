import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load dataset
file_path = "C:/Users/uragu/OneDrive/Desktop/Analog Sample/0.05% Nutrient conc.xlsx"
df = pd.read_excel(file_path, skiprows=1)  # Skip first row

df.columns = ["time", "cell_growth"]  # Rename columns

# Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[["cell_growth"]])

# Function to create sequences
def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(df_scaled, seq_length)

# Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Function to build and train LSTM model
def train_lstm_model(layers, units, dropout_rate, epochs=100, batch_size=8):
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units, return_sequences=(layers > 1), input_shape=(seq_length, 1)))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    
    # Additional LSTM layers (if applicable)
    for i in range(1, layers):
        return_seq = (i < layers - 1)
        model.add(LSTM(units, return_sequences=return_seq))
        if dropout_rate:
            model.add(Dropout(dropout_rate))
    
    # Dense output layer
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    
    return model, history

# Train 1-layer LSTM model
model_1_layer, history_1_layer = train_lstm_model(layers=1, units=50, dropout_rate=0.2, epochs=100)

# Train 3-layer LSTM model
model_3_layer, history_3_layer = train_lstm_model(layers=3, units=50, dropout_rate=0.2, epochs=100)

# Plot learning curves
plt.figure(figsize=(12, 6))

# 1-Layer LSTM Learning Curve
plt.subplot(1, 2, 1)
plt.plot(history_1_layer.history['loss'], label='Training Loss')
plt.plot(history_1_layer.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('1-Layer LSTM Learning Curve')
plt.legend()
plt.grid()

# 3-Layer LSTM Learning Curve
plt.subplot(1, 2, 2)
plt.plot(history_3_layer.history['loss'], label='Training Loss')
plt.plot(history_3_layer.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('3-Layer LSTM Learning Curve')
plt.legend()
plt.grid()

plt.show()
