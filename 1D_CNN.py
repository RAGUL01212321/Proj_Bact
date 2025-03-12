import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# ==== Generate Sample Data ====
# Create a random dataset with 100 samples, 20 time steps, and 1 feature
X = np.random.rand(100, 20, 1)  # Shape = (samples, time_steps, features)
y = np.random.rand(100, 1)  # Output shape = (samples, 1)

# ==== Build the 1D CNN Model ====
model = Sequential()

# Convolutional Layer
model.add(Conv1D(
    filters=32,              # Number of filters (try changing)
    kernel_size=3,           # Size of the kernel (try changing)
    strides=1,               # Step size (try changing)
    padding='same',          # 'same' or 'valid'
    activation='relu',       # Activation function
    input_shape=(20, 1)      # (time_steps, features)
))

# Pooling Layer
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

# Second Convolutional Layer
model.add(Conv1D(
    filters=32,
    kernel_size=6,
    strides=1,
    padding='same',
    activation='tanh'
))

# Second Pooling Layer
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

# Flatten the output
model.add(Flatten())

# Dense Layer
model.add(Dense(64, activation='relu'))

# Dropout Layer
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(1))  # Single value output (regression task)

# ==== Compile the Model ====
model.compile(optimizer='adam', loss='mse')

# ==== Model Summary ====
model.summary()

# ==== Train the Model ====
history = model.fit(X, y, epochs=20, batch_size=16, validation_split=0.2, verbose=1)

# ==== Plot Learning Curve ====
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.title('Learning Curve - 1D CNN')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
