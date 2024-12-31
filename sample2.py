import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# 1. Simulate Tumor Growth
# -------------------------------

# Gompertz growth equation parameters
def gompertz_growth(t, x0, a, K):
    """
    Simulates tumor growth using the Gompertz equation.
    dx/dt = a * x(t) * ln(K / x(t))
    
    Parameters:
        t: Time (array)
        x0: Initial tumor size
        a: Growth rate constant
        K: Carrying capacity (max tumor size)
    Returns:
        Tumor size (x) at each time step
    """
    x = np.zeros_like(t)
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i - 1] + a * x[i - 1] * np.log(K / x[i - 1]) * (t[i] - t[i - 1])
    return x

# Parameters
time_steps = np.linspace(0, 50, 500)  # Time range (0 to 50 days, 500 points)
x0 = 0.1                              # Initial tumor size
a = 0.1                               # Growth rate constant
K = 10.0                              # Carrying capacity

# Simulate tumor growth
tumor_sizes = gompertz_growth(time_steps, x0, a, K)

# Plot the simulated tumor growth
plt.figure(figsize=(10, 6))
plt.plot(time_steps, tumor_sizes, label="Tumor Size (Gompertz Model)", color='b')
plt.title("Simulated Tumor Growth Using Gompertz Model")
plt.xlabel("Time (days)")
plt.ylabel("Tumor Size")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# 2. Prepare Dataset for LSTM
# -------------------------------

# Normalize the tumor size data
scaler = MinMaxScaler()
tumor_sizes_scaled = scaler.fit_transform(tumor_sizes.reshape(-1, 1))

# Create time-series dataset
def create_time_series(data, window_size):
    """
    Creates time-series data for LSTM.
    
    Parameters:
        data: Scaled tumor size data
        window_size: Number of time steps to use for prediction
    Returns:
        X: Features (time-series input)
        y: Targets (next tumor size value)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 20  # Number of time steps to use for prediction
X, y = create_time_series(tumor_sizes_scaled, window_size)

# Split into training and testing datasets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# -------------------------------
# 3. Build LSTM Model
# -------------------------------

# LSTM model architecture
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(window_size, 1), return_sequences=True),
    LSTM(50, activation='tanh'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test),
    epochs=100, 
    batch_size=32, 
    callbacks=[early_stopping],
    verbose=1
)

# -------------------------------
# 4. Evaluate the Model
# -------------------------------

# Predict tumor sizes
y_pred = model.predict(X_test)

# Rescale predictions back to original scale
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test_rescaled)), y_test_rescaled, label="Actual Tumor Size", color='b')
plt.plot(range(len(y_pred_rescaled)), y_pred_rescaled, label="Predicted Tumor Size", color='r')
plt.title("Tumor Growth Prediction with LSTM")
plt.xlabel("Time Steps")
plt.ylabel("Tumor Size")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# 5. Save the Model and Outputs
# -------------------------------

# Save the model
model.save("tumor_growth_lstm_model.h5")
print("Model saved as tumor_growth_lstm_model.h5")

# Save simulated data
np.savetxt("simulated_tumor_growth.csv", np.column_stack((time_steps, tumor_sizes)), delimiter=",", header="Time,TumorSize", comments='')
print("Simulated tumor growth data saved as simulated_tumor_growth.csv")
