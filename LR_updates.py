import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path, skiprows=2, names=["time", "cell_growth"])
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[["cell_growth"]])
    return df_scaled, scaler

def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_and_train_model(X_train, y_train, X_test, y_test, lstm_units, dropout_rate, learning_rate, epochs=100, batch_size=8):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    return history

def plot_learning_curves(histories, labels):
    plt.figure(figsize=(12, 8))
    for history, label in zip(histories, labels):
        plt.plot(history.history['loss'], label=f'Training Loss - {label}')
        plt.plot(history.history['val_loss'], label=f'Validation Loss - {label}', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Learning Curves for Different Hyperparameters')
    plt.legend()
    plt.grid()
    plt.show()

# Load dataset
file_path = "C:/Users/uragu/OneDrive/Desktop/Analog Sample/0.05% Nutrient conc.xlsx"
data_scaled, scaler = load_and_preprocess_data(file_path)

# Create sequences
seq_length = 5
X, y = create_sequences(data_scaled, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define hyperparameter settings to test
hyperparameter_settings = [
    (10, 0.2, 1.0),  # Low LSTM units, standard dropout, standard LR
  # Low learning rate
]

histories = []
labels = []

# Train models with different hyperparameters
for lstm_units, dropout_rate, learning_rate in hyperparameter_settings:
    history = build_and_train_model(X_train, y_train, X_test, y_test, lstm_units, dropout_rate, learning_rate)
    histories.append(history)
    labels.append(f'LSTM {lstm_units}, Dropout {dropout_rate}, LR {learning_rate}')

# Plot all learning curves
plot_learning_curves(histories, labels)
