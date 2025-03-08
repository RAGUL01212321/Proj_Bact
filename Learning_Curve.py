import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Load dataset
file_path = r"C:\Users\uragu\OneDrive\Desktop\Analog Sample\0.05% Nutrient conc.xlsx"
df = pd.read_excel(file_path, skiprows=2)  # Skip first two rows (header + nutrient content)
time = df.iloc[:, 0].values  # Extract time column
cells = df.iloc[:, 1].values  # Extract cell population column

# Normalize data
scaler = MinMaxScaler()
cells_scaled = scaler.fit_transform(cells.reshape(-1, 1))

# Convert data into sequences for LSTM
sequence_length = 50  # 50 time steps per sample
X, y = [], []
for i in range(len(cells_scaled) - sequence_length):
    X.append(cells_scaled[i:i+sequence_length])
    y.append(cells_scaled[i+sequence_length])

X, y = np.array(X), np.array(y)

# Split into train/test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Function to create and train LSTM model
def train_lstm(learning_rate):
    model = Sequential([
        LSTM(3, input_shape=(sequence_length, 1)),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    return history.history['loss']

# Test different learning rates
learning_rates = [0.0001]
loss_curves = {}

for lr in learning_rates:
    loss_curves[lr] = train_lstm(lr)

# Plot the loss curves
plt.figure(figsize=(8, 5))
for lr, loss in loss_curves.items():
    plt.plot(loss, label=f"LR={lr}")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curves for Different Learning Rates")
plt.legend()
plt.grid(True)
plt.show()
