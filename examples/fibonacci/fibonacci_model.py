# Library imports
import melpy.NeuralNetworks as nn
from melpy.tensor import *
from melpy.preprocessing import generate_sequence_dataset

# Function definitions
def fibonacci(n: int):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Dataset creation
data = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
inputs, targets = generate_sequence_dataset(data, 2)

inputs = Tensor(inputs.reshape(*inputs.shape, 1))
targets = Tensor(targets.reshape(-1, 1))

# Model creation
model = nn.Sequential(inputs, targets)

model.add(nn.LSTM(inputs.shape[2], 64, activation="relu", num_layers=1))
model.add(nn.Dense(64, targets.shape[1]))

model.compile(nn.MeanSquaredError(), optimizer=nn.Adam(learning_rate=0.01))
model.summary()

# Model training
epochs = 200
model.fit(epochs=epochs, verbose=1)
model.results()

# Model testing
test_data = Tensor([144, 233, 377, 610, 987]).reshape(1, 5, 1)
predicted_fib = model.predict(test_data)

print("\nThe predicted next fibonacci number (the 17th): ", predicted_fib)
print("The actual 17th fibonacci number:", fibonacci(17))
print("Difference is", abs(predicted_fib - fibonacci(17)))