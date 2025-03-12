import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, MaxPooling1D, Flatten

# Load the dataset
file_path = r'C:\Users\uragu\OneDrive\Desktop\Analog Sample\datasets\nutrient_0.01.csv'
data = pd.read_csv(file_path)

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Population']])

# Prepare the data for the model
X, y = [], []
time_steps = 10
for i in range(len(scaled_data) - time_steps):
    X.append(scaled_data[i:i + time_steps])
    y.append(scaled_data[i + time_steps])

X, y = np.array(X), np.array(y)

# Function to build and train models
def build_and_train(model_type):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(LSTM(32))
    elif model_type == '1D CNN':
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(Flatten())
    elif model_type == 'GRU':
        model.add(GRU(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(GRU(32))
    
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2, verbose=0)
    
    return model, history

# Train all three models
models = {}
histories = {}
for model_type in ['LSTM', '1D CNN', 'GRU']:
    print(f'Training {model_type} model...')
    model, history = build_and_train(model_type)
    models[model_type] = model
    histories[model_type] = history

# Predict and rescale
predictions = {}
for model_type in models:
    predicted = models[model_type].predict(X)
    predictions[model_type] = scaler.inverse_transform(predicted)

actual = scaler.inverse_transform(y.reshape(-1, 1))

# Calculate accuracy metrics
accuracy_results = {}
for model_type in predictions:
    mae = mean_absolute_error(actual, predictions[model_type])
    r2 = r2_score(actual, predictions[model_type])
    accuracy_results[model_type] = {'MAE': mae, 'R²': r2}

# Display accuracy comparison
print("\nModel Performance:")
for model_type, result in accuracy_results.items():
    print(f"{model_type}: MAE = {result['MAE']:.4f}, R² = {result['R²']:.4f}")

# Plot predicted vs actual data
plt.figure(figsize=(12, 8))
for model_type in predictions:
    plt.plot(predictions[model_type], label=f'{model_type} Prediction')
plt.plot(actual, label='Actual Data', color='black', linewidth=1.5)
plt.title('Predicted vs Actual Data')
plt.legend()
plt.show()

# Plot learning curves
plt.figure(figsize=(12, 8))
for model_type in histories:
    plt.plot(histories[model_type].history['loss'], label=f'{model_type} Training Loss')
    plt.plot(histories[model_type].history['val_loss'], label=f'{model_type} Validation Loss', linestyle='--')
plt.title('Learning Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))

for model_type in histories:
    plt.plot(histories[model_type].history['loss'], label=f'{model_type} Training Loss')
    plt.plot(histories[model_type].history['val_loss'], label=f'{model_type} Validation Loss', linestyle='--')

plt.title('Learning Curves for LSTM, GRU, and 1D CNN')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
