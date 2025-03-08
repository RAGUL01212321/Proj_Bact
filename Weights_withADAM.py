import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# ✅ Load and prepare data (same as before)
file_path = "C:/Users/uragu/OneDrive/Desktop/Analog Sample/0.05% Nutrient conc.xlsx"
df = pd.read_excel(file_path, skiprows=1)
df.columns = ["time", "cell_growth"]

# ✅ Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[["cell_growth"]])

# ✅ Prepare sequences
def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(df_scaled, seq_length)

# ✅ Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# ✅ Split into train & test sets (80% train, 20% test)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ✅ Build Single-Unit LSTM Model
model = Sequential([
    LSTM(1, input_shape=(seq_length, 1)),  # **Single LSTM Unit**
    Dense(1)  
])

# ✅ Compile with Adam Optimizer
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# ✅ Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=1)

# ✅ Retrieve LSTM Weights
weights = model.layers[0].get_weights()

# ✅ Split Weights into Gates (Since We Have Only 1 LSTM Unit)
W_x = weights[0]  # Input weight matrix (shape: [input_dim, 4])
W_h = weights[1]  # Recurrent weight matrix (shape: [units, 4])
b = weights[2]    # Bias vector (shape: [4])

# ✅ Extract Weights for Each Gate
units = 1  # Since we only have **one** LSTM unit
gate_names = ["Input Gate (i)", "Forget Gate (f)", "Output Gate (o)", "Cell State (g)"]

# ✅ Reshape and Extract Weights for Each Gate
W_x_split = np.split(W_x, 4, axis=1)  # Split into 4 gates along columns
W_h_split = np.split(W_h, 4, axis=1)  # Split into 4 gates along columns
b_split = np.split(b, 4)  # Split into 4 biases

# ✅ Display the Weights for Each Gate
for i, gate in enumerate(gate_names):
    print(f"\n🔹 {gate} Weights:")
    print(f"  - W_x: {W_x_split[i].flatten()}")
    print(f"  - W_h: {W_h_split[i].flatten()}")
    print(f"  - Bias: {b_split[i].flatten()}")

# ✅ Plot Learning Curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('LSTM Learning Curve')
plt.legend()
plt.grid()
plt.show()
