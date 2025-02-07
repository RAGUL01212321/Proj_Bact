import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load the dataset
file_path = "C:/Users/uragu/OneDrive/Desktop/Analog Sample/0.05% Nutrient conc.xlsx"
df = pd.read_excel(file_path, skiprows=1)  # Skip the first row

# Rename columns
df.columns = ["time", "cell_growth"]

# Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[["cell_growth"]])

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Hyperparameters
seq_length = 5  # Length of input sequences
lstm_units = 50  # Number of LSTM units per layer
batch_size = 8   # Batch size for training
epochs = 50      # Number of training epochs
learning_rate = 0.01  # Learning rate

# Prepare data
X, y = create_sequences(df_scaled, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into training & testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
def build_lstm_model(lstm_units, learning_rate):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(lstm_units, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Train model
model = build_lstm_model(lstm_units, learning_rate)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)

# Plot learning curves
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], linestyle='dashed', label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Learning Curve')
plt.legend()
plt.grid()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load the dataset
file_path = "C:/Users/uragu/OneDrive/Desktop/Analog Sample/0.05% Nutrient conc.xlsx"
df = pd.read_excel(file_path, skiprows=1)  # Skip the first row

# Rename columns
df.columns = ["time", "cell_growth"]

# Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[["cell_growth"]])

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Hyperparameters
seq_length = 5  # Length of input sequences
lstm_units = 50  # Number of LSTM units per layer
batch_size = 8   # Batch size for training
epochs = 50      # Number of training epochs
learning_rate = 0.01  # Learning rate

# Prepare data
X, y = create_sequences(df_scaled, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into training & testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
def build_lstm_model(lstm_units, learning_rate):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(lstm_units, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Train model
model = build_lstm_model(lstm_units, learning_rate)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)

# Plot learning curves
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], linestyle='dashed', label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Learning Curve')
plt.legend()
plt.grid()
plt.show()
