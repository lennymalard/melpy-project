# %% Data loading
from keras.datasets import mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train[:, np.newaxis, :, :]/255
X_test = X_test[:, np.newaxis, :, :]/255

# %% Data preprocessing
from melpy.preprocessing import StandardScaler, OneHotEncoder

sc = StandardScaler()
X_train_reshaped = X_train.reshape(X_train.shape[0],-1)
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

for i in range(1, n_samples+1):
    rand_n = randint(0, X_train.shape[0] - 1)
    plt.subplot(1, n_samples, i)
    plt.title(np.argmax(y_train[rand_n]))
    plt.imshow(X_train[rand_n, 0,  :, :], cmap="gray")

#%% Modeling
import melpy.NeuralNetworks as nn

model = nn.Sequential(X_train, y_train, X_test, y_test)

model.add(nn.Convolution2D(in_channels=1, out_channels=32, kernel_size=2, padding="same", weight_initializer="glorot_uniform", activation="leaky_relu"))
model.add(nn.Pooling2D(pool_size=2, stride=2, mode="max"))
model.add(nn.Convolution2D(in_channels=32, out_channels=64, kernel_size=2, padding="same", weight_initializer="glorot_uniform" , activation="leaky_relu"))
model.add(nn.Flatten())
model.add(nn.Dense(model.get_flatten_length(), 128, activation="leaky_relu"))
model.add(nn.Dense(128, 10, weight_initializer="glorot_normal", activation="softmax"))

model.compile(optimizer= nn.Adam(learning_rate = 1e-3), loss_function=nn.CategoricalCrossEntropy())
model.summary()

# %% Training
model.fit(epochs=5, batch_size=256, verbose=2)
model.results()

# %% Save
model.save_params("mnist_parameters")
model.save_histories("mnist_history")

# %% Test
predictions = model.predict(X_test[:1000])
print(f"\nTest Accuracy: {nn.accuracy(y_test[:1000], predictions)}")

# %% Random predictions
n_samples = 5

figure = plt.figure()

for i in range(1, n_samples+1):
    rand_n = randint(0, predictions.shape[0] - 1)
    plt.subplot(1, n_samples, i)
    plt.title(f"pred: {np.argmax(predictions[rand_n])}, target: {np.argmax(y_test[rand_n])}")
    plt.imshow(X_test[rand_n, 0,  :, :], cmap="gray")

# %% Feature maps visualization
figure = plt.figure()

for i in range(1,10,3):
    rand_channel1 = randint(0, model.layers[0].outputs.shape[1] - 1)
    rand_channel2 = randint(0, model.layers[1].outputs.shape[1] - 1)

    rand_n = randint(0, predictions.shape[0] - 1)

    plt.subplot(3, 3, i)
    plt.title("input")
    plt.imshow(model.layers[0].inputs.array[rand_n, 0,  :, :], cmap="gray")

    plt.subplot(3, 3, i+1)
    plt.title("conv2d")
    plt.imshow(model.layers[0].outputs.array[rand_n, rand_channel1,  :, :], cmap="gray")

    plt.subplot(3, 3, i+2)
    plt.title("pooling2d")
    plt.imshow(model.layers[1].outputs.array[rand_n, rand_channel2,  :, :], cmap="gray")
