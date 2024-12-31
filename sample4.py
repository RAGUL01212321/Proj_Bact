import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# --- Step 1: Simulate bacterial growth data (without drug) ---
time_steps = 100  # Number of time steps
growth_rate = 0.1  # Growth rate (per hour)
initial_population = 1  # Initial bacterial population

# Generate time series data (logistic growth model without drug)
time = np.arange(0, time_steps)
bacterial_population = initial_population * np.exp(growth_rate * time)

# Add some noise to simulate realistic biological variations
noise = np.random.normal(0, 0.05, bacterial_population.shape)
bacterial_population_noisy = bacterial_population + noise

# --- Step 2: Simulate drug effect ---
# Simulate drug effect by reducing growth at certain time points
drug_effect_start = 40  # Time when drug is introduced
drug_effect_end = 80   # Time when drug effect ends
drug_effect_factor = 0.5  # 50% reduction in growth

# Apply drug effect to the bacterial population
bacterial_population_with_drug = bacterial_population_noisy.copy()
bacterial_population_with_drug[drug_effect_start:drug_effect_end] *= drug_effect_factor

# --- Step 3: Visualize the generated data ---
plt.plot(time, bacterial_population_noisy, label='Bacterial Growth (No Drug)')
plt.plot(time, bacterial_population_with_drug, label='Bacterial Growth (With Drug)', linestyle='--')
plt.xlabel('Time (hours)')
plt.ylabel('Bacterial Population')
plt.legend()
plt.show()

# --- Step 4: Prepare data for RNN ---
# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(bacterial_population_with_drug.reshape(-1, 1))

# Reshape data for RNN (samples, timesteps, features)
def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Create dataset with time_steps=10
time_steps = 10
X, y = create_dataset(data_normalized, time_steps)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape for RNN input: [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# --- Step 5: Build and Train the RNN model ---
# Build the RNN model
model = Sequential()
model.add(SimpleRNN(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# --- Step 6: Make Predictions and Visualize Results ---
y_pred = model.predict(X_test)

# Inverse transform the predictions back to original scale
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the predictions vs actual data
plt.plot(y_test_rescaled, label='Actual Bacterial Growth (With Drug)')
plt.plot(y_pred_rescaled, label='Predicted Bacterial Growth (With Drug)', linestyle='--')
plt.legend()
plt.show()
