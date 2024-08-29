import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# Load data
with h5py.File('flow_past_cylinder.h5', 'r') as f:
    velocity_data = f['velocity'][:]
    density_data = f['density'][:]
    boundary_data = f['boundary'][:]

# Combine inputs: velocity, density, and boundary
X = np.concatenate((velocity_data, density_data, boundary_data), axis=-1).reshape(velocity_data.shape[0], -1)
# Targets: velocity and density
y = np.concatenate((velocity_data, density_data), axis=-1).reshape(velocity_data.shape[0], -1)

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Define the CNN architecture
model = models.Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(velocity_data.shape[1], velocity_data.shape[2], 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(y_train.shape[1] * y_train.shape[2], activation='linear'),  # Output layer
    
    # Reshape the output to the desired shape
    layers.Reshape((y_train.shape[1], y_train.shape[2], y_train.shape[3]))  # Adjust based on your target shape
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    validation_data=(X_val, y_val),
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

model.save('velocity_density_predictor.h5')