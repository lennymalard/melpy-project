# %% Data loading
import numpy as np
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

train_set_size = X_train.shape[0]

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

# %% Modeling
import melpy.NeuralNetworks as nn

model = nn.Sequential(input_shape=(1, 1, 28, 28))

model.add(nn.Convolution2D(in_channels=1, out_channels=32, kernel_size=2, padding="same", activation="leaky_relu"))
model.add(nn.Pooling2D(pool_size=2, stride=2, mode="max"))
model.add(nn.Convolution2D(in_channels=32, out_channels=64, kernel_size=2, padding="same", activation="leaky_relu"))
model.add(nn.Flatten())
model.add(nn.Dense(model.get_flatten_length(), 128, activation="leaky_relu"))
model.add(nn.Dense(128, 10, activation="softmax"))

model.summary()

# %% Parameters loading
model.load_params("results/mnist_parameters_01_24_2025-19_07_07.h5")

# %% Test
from melpy.metrics import accuracy

predictions = model.predict(X_test[:1000])
print(f"\nTest Accuracy: {nn.accuracy(y_test[:1000], predictions)}")

# %% App
import gradio as gr
import cv2

default_canvas = np.ones((800, 800, 3), dtype=np.uint8) * 255

def predict(image_dict):
    image = image_dict["composite"]
    image = image.astype(np.uint8)

    if image.ndim == 3 and image.shape[-1] == 3:
        image = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)).astype(np.float64)

    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 1, 28, 28) / 255.0
    image = sc.transform(image.reshape(1, -1)).reshape(1, 1, 28, 28)

    prediction = model.predict(image)
    return {i: prediction[:,i] for i in range(10)}

def clear_canvas():
    return default_canvas  # Reset the canvas

with gr.Blocks() as iface:
    sketchpad = gr.Sketchpad(
        label="Digit",
        type="numpy",
        image_mode="RGB",
        value=default_canvas,
        layers=False,
        brush= gr.Brush(default_color="black", default_size=25)
    )
    gr.Textbox(label="Note", value="Ensure that the digit is centered and occupies sufficient space for proper recognition. "
                                   "Also, note that MNIST does not perform well in real-world scenarios, so some confusion from the model is expected.")
    clear_button = gr.Button("Clear")
    output = gr.Label(num_top_classes=5)

    sketchpad.change(predict, inputs=sketchpad, outputs=output)
    clear_button.click(clear_canvas, outputs=sketchpad)

if __name__ == "__main__":
    iface.launch(share=True)
