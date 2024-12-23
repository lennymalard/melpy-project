# %% Data loading
from keras.datasets import mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

n_samples = X_train.shape[0]

X_train = X_train[:n_samples, np.newaxis, :, :] / 255
y_train = y_train[:n_samples]

X_test = X_test[:, np.newaxis, :, :] / 255

# %% Data preprocessing
from melpy.preprocessing import StandardScaler, OneHotEncoder

sc = StandardScaler()
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_train = sc.transform(X_train_reshaped).reshape(X_train.shape)

X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
X_test = sc.transform(X_test_reshaped).reshape(X_test.shape)

ohe = OneHotEncoder()
y_train = ohe.transform(y_train)
y_test = ohe.transform(y_test)

# %% Data visualization
import matplotlib.pyplot as plt
from random import randint

n_samples = 5

figure = plt.figure()

for i in range(1, n_samples + 1):
    rand_n = randint(0, X_train.shape[0] - 1)
    plt.subplot(1, n_samples, i)
    plt.title(np.argmax(y_train[rand_n]))
    plt.imshow(X_train[rand_n, 0, :, :], cmap="gray")
# %% Modeling
import melpy.NeuralNetworks as nn

initializer = "he_uniform"

model = nn.Sequential(X_train, y_train, X_test, y_test)

model.add(nn.Convolution2D(in_channels=1, out_channels=32, kernel_size=2, padding="same", activation=nn.LeakyReLU()))
model.add(nn.Pooling2D(pool_size=2, stride=2, mode="max"))
model.add(nn.Convolution2D(in_channels=32, out_channels=64, kernel_size=2, padding="same", activation=nn.LeakyReLU()))
model.add(nn.Flatten())
model.add(nn.Dense(model.get_flatten_length(), 128, weight_initializer = initializer, activation=nn.LeakyReLU()))
model.add(nn.Dense(128,64, activation=nn.LeakyReLU()))
model.add(nn.Dense(64,10, activation=nn.LeakyReLU()))
model.add(nn.Dense(10, 10, weight_initializer="glorot_normal", activation=nn.Softmax()))

model.compile(optimizer=nn.Adam(learning_rate=1e-3), cost_function=nn.CategoricalCrossEntropy())
model.summary()
# %% Parameters loading
model.load_params("results/mnist_parameters_12_06_2024-19_53_11.pkl") # NEEDS UPDATE

# %% Test
n_test_samples = 1000

predictions = model.predict(X_test[:n_test_samples])
print(f"\nTest Accuracy: {nn.accuracy(y_test[:n_test_samples], predictions)}")

# %% Random predictions
n_samples = 5

figure = plt.figure()

for i in range(1, n_samples + 1):
    rand_n = randint(0, n_test_samples - 1)
    plt.subplot(1, n_samples, i)
    plt.title(f"pred: {np.argmax(predictions[rand_n])}, target: {np.argmax(y_test[rand_n])}")
    plt.imshow(X_test[rand_n, 0, :, :], cmap="gray")

# %% Feature maps visualization
figure = plt.figure()

for i in range(1, 10, 3):
    rand_channel1 = randint(0, model.train_layers[0].outputs.shape[1] - 1)
    rand_channel2 = randint(0, model.train_layers[1].outputs.shape[1] - 1)

    rand_n = randint(0, n_test_samples - 1)

    plt.subplot(3, 3, i)
    plt.title("input")
    plt.imshow(model.train_layers[0].inputs[rand_n, 0, :, :], cmap="gray")

    plt.subplot(3, 3, i + 1)
    plt.title("conv2d")
    plt.imshow(model.train_layers[1].outputs[rand_n, rand_channel1, :, :], cmap="gray")

    plt.subplot(3, 3, i + 2)
    plt.title("pooling2d")
    plt.imshow(model.train_layers[2].outputs[rand_n, rand_channel2, :, :], cmap="gray")