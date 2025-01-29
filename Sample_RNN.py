import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

# Load the data from the Excel file with the correct header row
df = pd.read_excel('0.05% Nutrient conc.xlsx', header=1)  # Skips the first row and uses the second row as header

# Print column names to inspect them
print("Original Columns:", df.columns)

# Clean column names by stripping leading/trailing spaces
df.columns = df.columns.str.strip()

# Print the cleaned column names
print("Cleaned Columns:", df.columns)

# Now, access the 'time' and 'cell_growth' columns
time_data = df['time']  # Time in minutes
cell_growth_data = df['cell_growth']  # Cell growth in cells/ml

# Check for any missing values and handle them
if df.isnull().values.any():
    print("Missing values found, filling with zeros.")
    df = df.fillna(0)  # You can use other methods to handle missing data as well

# Convert the data to a numpy array
time_data = np.array(time_data).reshape(-1, 1)
cell_growth_data = np.array(cell_growth_data).reshape(-1, 1)

# Normalize the data to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_time = scaler.fit_transform(time_data)
scaled_growth = scaler.fit_transform(cell_growth_data)

# Prepare the data for RNN
def create_dataset(time, growth, time_step=1):
    X, y = [], []
    for i in range(len(time) - time_step):
        X.append(time[i:i+time_step, 0])
        y.append(growth[i + time_step, 0])
    return np.array(X), np.array(y)

# Reshape data for LSTM (samples, time steps, features)
time_step = 10  # Number of previous time steps to consider
X, y = create_dataset(scaled_time, scaled_growth, time_step)

# Reshaping X to be in the form (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the RNN model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Evaluate the model
predictions = model.predict(X_test)

# Inverse scaling to get the original values
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))  # Reshape to 2D before inverse transform
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))  # Reshape to 2D before inverse transform

# Print the first few predictions and actual values
print(f"Predictions: {predictions[:5]}")
print(f"Actual Values: {y_test[:5]}")


# Print the first few predictions and actual values
print(f"Predictions: {predictions[:5]}")
print(f"Actual Values: {y_test[:5]}")
