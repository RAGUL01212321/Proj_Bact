import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load the dataset
file_path = "C:/Users/uragu/OneDrive/Desktop/Analog Sample/0.05% Nutrient conc.xlsx"  # Adjusted for execution
df = pd.read_excel(file_path, skiprows=1)  # Skip the first row

# Rename columns
df.columns = ["time", "cell_growth"]

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

# Split into training & testing sets (80% train, 20% test)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define different learning rates to test
learning_rates = [1.0]
histories = {}

# Train models with different learning rates
for lr in learning_rates:
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')

    history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=0)
    histories[lr] = history

# Plot learning curves
plt.figure(figsize=(12, 8))
for lr, history in histories.items():
    plt.plot(history.history['loss'], label=f'Train Loss (LR={lr})')
    plt.plot(history.history['val_loss'], linestyle='dashed', label=f'Val Loss (LR={lr})')

plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Learning Curves for Different Learning Rates')
plt.legend()
plt.grid()
plt.show()
