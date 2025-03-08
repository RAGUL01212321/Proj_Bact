import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ✅ Load the dataset
file_path = "C:/Users/uragu/OneDrive/Desktop/Analog Sample/0.05% Nutrient conc.xlsx"
df = pd.read_excel(file_path, skiprows=1)  # Skip the first row (empty), keep column names

# ✅ Rename columns for easier access
df.columns = ["time", "cell_growth"]

# ✅ Normalize data (scaling between 0 and 1)
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[["cell_growth"]])

# ✅ Prepare input sequences (time-series format)
def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])  # Past `seq_length` values
        y.append(data[i+seq_length])    # Next value
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(df_scaled, seq_length)

# ✅ Reshape for LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# ✅ Split into training & testing sets (80% train, 20% test)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ✅ Build RNN model using LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),  
    LSTM(50, return_sequences=False), 
    Dense(25, activation='relu'),  
    Dense(1)  # Single output neuron
])

# ✅ Compile the model
model.compile(optimizer='adam', loss='mse')

# ✅ Train the model & store history
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)

# ✅ Plot Learning Curve (Loss vs. Epochs)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Learning Curve of RNN')
plt.legend()
plt.grid()
plt.show()
