# Library imports
import numpy as np
import melpy.NeuralNetworks as nn
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

# Load the Boston housing dataset.
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

boston_features = {
    'Average Number of Rooms': 5,  # Feature index for average number of rooms
}

# Extract only the desired feature (Average Number of Rooms)
X_train_1d = X_train[:, boston_features['Average Number of Rooms']]
X_test_1d = X_test[:, boston_features['Average Number of Rooms']]

# Scatter plot of the feature vs. the target value (median price)
plt.figure(figsize=(15, 5))

plt.xlabel('Average Number of Rooms')
plt.ylabel('Median Price [$K]')
plt.grid("on")
plt.scatter(X_train_1d[:], y_train, color='green', alpha=0.5);

# Initialize a simple model using 1 feature (rooms) as input
model = nn.Sequential(X_train_1d.reshape(-1, 1), y_train.reshape(-1, 1))

# Add a dense (fully connected) layer
model.add(nn.Dense(1, 1))  # 1 input, 1 output (price)
 
model.summary()  # Summary of the model structure

# Compile the model with optimizer and loss function
model.compile(optimizer=nn.Adam(learning_rate=0.05),
              loss_function=nn.MeanSquaredError())

# Train the model using the data
model.fit()

# Display training results (e.g., loss value)
model.results()

# Predict housing prices for a given number of rooms
x = np.array([3, 4, 5, 6, 7]).reshape(-1, 1)
y_pred = model.predict(x)

print()

# Display predictions
for idx in range(len(x)):
    print(f"Predicted price of a home with {x[idx, 0]} rooms: ${int(y_pred[idx, 0] * 10) / 10}K")
