import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ✅ Load dataset
file_path = "C:/Users/uragu/OneDrive/Desktop/Analog Sample/0.05% Nutrient conc.xlsx"
df = pd.read_excel(file_path, skiprows=1)
df.columns = ["time", "cell_growth"]

# ✅ Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[["cell_growth"]])

# ✅ Function to create sequences
def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(df_scaled, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# ✅ Split into training & testing sets (80% train, 20% test)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ✅ Experiment with different hyperparameters
learning_rates = [0.0005, 0.001, 0.005]
batch_sizes = [8, 16, 32]
dropout_rates = [0.1, 0.2, 0.3]

results = {}

for lr in learning_rates:
    for batch_size in batch_sizes:
        for dropout in dropout_rates:
            # ✅ Build model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),  
                Dropout(dropout),  # Apply dropout
                LSTM(50, return_sequences=False),
                Dropout(dropout),
                Dense(25, activation='relu'),
                Dense(1)
            ])

            optimizer = Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss='mse')

            # ✅ Train model
            history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)

            # ✅ Store results
            results[(lr, batch_size, dropout)] = history.history

            # ✅ Plot learning curve
            plt.plot(history.history['loss'], label=f'Train (LR={lr}, BS={batch_size}, Drop={dropout})')
            plt.plot(history.history['val_loss'], label=f'Validation (LR={lr}, BS={batch_size}, Drop={dropout})')

plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Learning Curve for Different Hyperparameters')
plt.legend()
plt.grid()
plt.show()

# ✅ Find the best configuration
best_config = min(results, key=lambda k: min(results[k]['val_loss']))
print(f"Best Hyperparameter Set: Learning Rate={best_config[0]}, Batch Size={best_config[1]}, Dropout={best_config[2]}")
