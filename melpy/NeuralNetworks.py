import numpy as np
from .layers import *
from .losses import *
from .metrics import *
from .optimizers import *
from .callbacks import *
from .preprocessing import *
from .tensor import *
from math import sqrt
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
import h5py
import sys
from datetime import datetime
import os

class Sequential:
    """
    A sequential neural network model supporting various layer types including dense,
    convolutional, pooling, recurrent (LSTM), and utility layers (dropout, flatten).

    Attributes
    ----------
    train_inputs : Tensor
        Training input data (automatically converted to Tensor if numpy array is provided).
    train_targets : Tensor
        Training target labels/values (automatically converted to Tensor if numpy array is provided).
    val_inputs : Tensor or None
        Validation input data if provided. Default is None.
    val_targets : Tensor or None
        Validation targets if provided. Default is None.
    train_layers : list
        Layers used during training phase.
    val_layers : list
        Copy of training layers used for validation (if validation data provided).
    loss_function : Loss
        Configured loss function for model training.
    optimizer : Optimizer
        Optimization algorithm for parameter updates.
    train_loss_history : list
        Historical record of training loss values.
    val_loss_history : list
        Historical record of validation loss values (if validation enabled).
    train_accuracy_history : list
        Historical record of training accuracy values.
    val_accuracy_history : list
        Historical record of validation accuracy values (if validation enabled).
    runtime : float
        Total training duration in seconds.
    validation : bool
        Flag indicating if validation data was provided.

    Methods
    -------
    add(layer)
        Add a new layer to the model architecture.
    compile(loss_function, optimizer)
        Configure training parameters including loss and optimizer.
    fit(epochs, batch_size, verbose, callbacks, get_output)
        Train the model on provided data.
    predict(X)
        Generate predictions for input data.
    evaluate()
        Calculate loss and accuracy metrics on current parameters.
    save_params(filename)
        Save model parameters to HDF5 file.
    load_params(path)
        Load parameters from HDF5 file.
    save_histories(filename, extension)
        Save training metrics history to file.
    summary()
        Print model architecture summary.
    """

    def __init__(self, train_inputs, train_targets, val_inputs=None, val_targets=None):
        """
        Initialize model with training data and optional validation data.

        Parameters
        ----------
        train_inputs : np.ndarray or Tensor
            Input data for training. Shape (n_samples, ...)
        train_targets : np.ndarray or Tensor
            Target values for training. Shape (n_samples, ...)
        val_inputs : np.ndarray or Tensor, optional
            Input data for validation. Default None.
        val_targets : np.ndarray or Tensor, optional
            Target values for validation. Default None.

        Raises
        ------
        TypeError
            If inputs are not numpy arrays or Tensors
        """

        if not isinstance(train_inputs, np.ndarray) and not isinstance(train_inputs, Tensor):
            raise TypeError('`train_inputs` must be of type ndarray or Tensor.')
        if not isinstance(train_targets, np.ndarray) and not isinstance(train_targets, Tensor):
            raise TypeError('`train_targets` must be of type ndarray or Tensor.')
        if val_inputs is not None and val_targets is not None:
            if not isinstance(val_inputs, np.ndarray) and not isinstance(val_inputs, Tensor):
                raise TypeError('`val_targets` must be of type ndarray or Tensor.')
            if not isinstance(val_targets, np.ndarray) and not isinstance(val_targets, Tensor):
                raise TypeError('`val_targets` must be of type ndarray or Tensor.')

        self.train_inputs = Tensor(train_inputs, requires_grad=True) if isinstance(train_inputs, np.ndarray) else train_inputs
        self.train_input_batch = None
        self.train_targets = Tensor(train_targets, requires_grad=True) if isinstance(train_targets, np.ndarray) else train_targets
        self.train_target_batch = None
        self.train_outputs = None
        self.train_output_batch = None

        self.val_inputs = Tensor(val_inputs, requires_grad=True) if isinstance(val_inputs, np.ndarray) else val_inputs
        self.val_input_batch = None
        self.val_targets = Tensor(val_targets, requires_grad=True) if isinstance(val_targets, np.ndarray) else val_targets
        self.val_target_batch = None
        self.val_outputs = None
        self.val_output_batch = None

        self.batch_size = None
        self.val_batch_size = None

        self.predictions = None

        self.train_layers = []
        self.val_layers = []

        self.train_loss = 0.0
        self.train_loss_batch = 0.0
        self.val_loss = 0.0
        self.val_loss_batch = 0.0
        self.train_loss_history = []
        self.val_loss_history = []

        self.train_accuracy = 0.0
        self.val_accuracy = 0.0
        self.train_accuracy_history = []
        self.val_accuracy_history = []

        self.loss_function = None
        self.optimizer = None

        self.__is_trained__ = False
        self.__is_compiled__ = False
        self.runtime = 0.0
        self.validation = False

        if self.val_inputs is not None and self.val_targets is not None:
            self.validation = True

    def get_flatten_length(self):
        """
        Calculate output dimension after flatten operation.

        Returns
        -------
        int
            Flattened dimension size

        Raises
        ------
        ValueError
            If no Flatten layer exists in the model
        """
        self.train_layers[0].inputs = Tensor(self.train_inputs.array[0].reshape(1, *self.train_inputs.array[0].shape), requires_grad=True)
        for i in range(len(self.train_layers)):
            if isinstance(self.train_layers[i], Flatten):
                self.train_layers[i].forward()
                self.train_layers[0].inputs = self.train_inputs
                return self.train_layers[i].outputs.shape[1]
            elif i == len(self.train_layers) - 1:
                self.train_layers[0].inputs = self.train_inputs
                raise ValueError("There is no `Flatten` layer.")
            else:
                self.train_layers[i + 1].inputs = self.train_layers[i].forward()

    def add(self, layer):
        """
        Add a layer to the model architecture.

        Parameters
        ----------
        layer : Layer
            Must be one of: Dense, Convolution2D, Pooling2D,
            Flatten, Dropout, LSTM, Embedding

        Raises
        ------
        TypeError
            If layer is not of supported type
        """
        if not isinstance(layer, (Dense, Convolution2D, Pooling2D, Flatten, Dropout, LSTM, Embedding)):
            raise TypeError("`layer` must be of type `Dense`, `Convolution2D`, `Pooling2D`, `LSTM`, `Embedding`, `Flatten` or `Dropout`.")

        self.train_layers.append(layer)

    def forward(self):
        """
        Performs the forward pass through the layers of the architecture for both
        training and validation.

        This method computes the output for both the training and validation data.
        During training, the input batch goes through the layers in sequence,
        and the output is stored in `train_output_batch`. If validation data is
        provided, the validation input batch goes through the validation layers
        and the output is stored in `val_output_batch`.

        Returns
        -------
        None
        """
        self.train_layers[0].inputs = Tensor(self.train_input_batch.array, requires_grad=True)
        if self.validation:
            self.val_layers[0].inputs = self.val_input_batch
        for i in range(len(self.train_layers)):
            if i + 1 == len(self.train_layers):
                self.train_output_batch = self.train_layers[i].forward()
                self.loss_function.forward(self.train_target_batch, self.train_output_batch)
            else:
                if isinstance(self.train_layers[i], Dropout):
                    self.train_layers[i].training = True
                self.train_layers[i + 1].inputs = self.train_layers[i].forward()
            if self.validation:
                if i + 1 == len(self.val_layers):
                    self.val_output_batch = self.val_layers[i].forward()
                else:
                    self.val_layers[i + 1].inputs = self.val_layers[i].forward()


    def predict(self, X):
        """
        Makes predictions for the given input `X` using the trained model.

        This method runs a forward pass with the provided input `X` and returns
        the predicted output based on the current state of the trained network.

        Be careful, you might encounter an error if the number of dimensions of `X`
        doesn't match the number of dimensions of `self.train_inputs`.

        Parameters
        ----------
        X : ndarray, Tensor
            The input data for which predictions are to be made.

        Returns
        -------
        predictions : ndarray
            The predicted output for the given input `X`.

        Raises
        ------
        TypeError
            If `X` is not of type ndarray or Tensor.
        """
        if not isinstance(X, np.ndarray) and not isinstance(X, Tensor):
            raise TypeError("`X` must be of type ndarray or Tensor.")
        if X.ndim != self.train_inputs.ndim:
            raise TypeError("`X` must have the same number of dimensions as `self.train_inputs`.")
        self.train_layers[0].inputs = Tensor(X, requires_grad=True) if isinstance(X, np.ndarray) else X
        for i in range(len(self.train_layers)):
            if isinstance(self.train_layers[i], Dropout):
                self.train_layers[i].training = False
            if i + 1 == len(self.train_layers):
                self.predictions = self.train_layers[i].forward()
                return self.predictions.array
            self.train_layers[i + 1].inputs = self.train_layers[i].forward()

    def backward(self):
        """
        Performs the backward pass through the layers to compute gradients.

        This method calculates the gradients of the loss function with respect to
        the parameters (weights and biases) of the network. The gradients are propagated backward through
        each layer.

        Returns
        -------
        None
        """
        self.dX = self.loss_function.backward()
        for layer in reversed(self.train_layers):
            self.dX = layer.backward(self.dX)

    def verbose(self, verbose, epoch, epochs, start_time):
        """
        Prints training and validation metrics during training at specified intervals.

        This method provides feedback during training based on the verbosity level
        chosen. It can print the loss and accuracy for both the training and validation
        datasets, and also the training time. The method adjusts the print frequency
        based on the `verbose` parameter.

        Parameters
        ----------
        verbose : int
            The verbosity level (0, 1, or 2).
            0 - No output.
            1 - Print at the end of the training.
            2 - Print at the end of each epoch.
        epoch : int
            The current epoch number.
        epochs : int
            The total number of epochs.
        start_time : float
            The starting time of training to calculate the runtime.

        Raises
        ------
        ValueError
            If the `verbose` parameter is not 0, 1, or 2.
        TypeError
            If `verbose`, `epochs`, or `start_time` are not of the correct type.

        Returns
        -------
        None
        """
        if not isinstance(verbose, int) and verbose is not None:
            raise TypeError('`verbose` must be of type int or None.')
        if not isinstance(epochs, int):
            raise TypeError('`epochs` must be of type int.')
        if not isinstance(start_time, float):
            raise TypeError('`start_time` must be of type float.')

        if verbose == 2:
            if epoch + 1 < epochs:
                if self.validation:
                    print(f"[TRAINING METRICS] train_loss: {np.around(self.train_loss, 5)} · "
                          f"train_accuracy: {np.around(self.train_accuracy, 5)}\n" +
                          f"[VALIDATION METRICS] val_loss: {np.around(self.val_loss, 5)} · "
                          f"val_accuracy: {np.around(self.val_accuracy, 5)}\n\n")
                else:
                    print(f"[TRAINING METRICS] train_loss: {np.around(self.train_loss, 5)} | " +
                          f"train_accuracy: {np.around(self.train_accuracy, 5)}\n\n")
            elif epoch + 1 == epochs:
                if self.validation:
                    self.runtime = time.time() - start_time
                    string1 = f"| [TRAINING METRICS] train_loss: {np.around(self.train_loss, 5)} · " + \
                              f"train_accuracy: {np.around(self.train_accuracy, 5)} |"
                    string2 = f"| [VALIDATION METRICS] val_loss: {np.around(self.val_loss, 5)} · " + \
                              f"val_accuracy: {np.around(self.val_accuracy, 5)} |"
                    string1_length = len(string1)
                    string2_length = len(string2)
                    print("\n" + string1_length * "-" + "\n" + string1 + "\n" +
                          string1_length * "-" + "\n" +
                          string2 + (string1_length - string2_length - 1) * " " + "\n" +
                          string1_length * "-")
                    print(f"{round(self.runtime, 5)} seconds")
                else:
                    self.runtime = time.time() - start_time
                    string = f"| [TRAINING METRICS] train_loss: {np.around(self.train_loss, 5)} · " + \
                             f"train_accuracy: {np.around(self.train_accuracy, 5)} |"
                    string_length = len(string)
                    print("\n" + string_length * "-" + "\n" + string + "\n" + string_length * "-")
                    print(f"{round(self.runtime, 5)} seconds")
        elif verbose == 1:
            if epoch + 1 == epochs:
                if self.validation:
                    string1 = f"| [TRAINING METRICS] train_loss: {np.around(self.train_loss, 5)} · " + \
                              f"train_accuracy: {np.around(self.train_accuracy, 5)} |"
                    string2 = f"| [VALIDATION METRICS] val_loss: {np.around(self.val_loss, 5)} · " + \
                              f"val_accuracy: {np.around(self.val_accuracy, 5)} |"
                    string1_length = len(string1)
                    string2_length = len(string2)
                    print("\n" + string1_length * "-" + "\n" + string1 + "\n" + string1_length * "-" + "\n" +
                          string2 + (string1_length - string2_length - 1) * " " + "\n" + string1_length * "-")
                else:
                    string = f"| [TRAINING METRICS] train_loss: {np.around(self.train_loss, 5)} · " + \
                             f"train_accuracy: {np.around(self.train_accuracy, 5)} |"
                    string_length = len(string)
                    print("\n" + string_length * "-" + "\n" + string + "\n" + string_length * "-")
        elif verbose == 0 or verbose is None:
            return
        else:
            raise ValueError("`verbose` must be 0, 1, or 2.")

    def compile(self, loss_function, optimizer=SGD(learning_rate=0.01)):
        """
        Compiles the model with the specified Loss function and optimizer.

        Parameters
        ----------
        loss_function : Loss
            The Loss function to be used for training.
        optimizer : Optimizer, optional
            The optimizer to be used for updating the model parameters. Default is SGD with a learning rate of 0.01.

        Raises
        ------
        ValueError
            If `loss_function` is not an instance of Loss.
            If `optimizer` is not an instance of Optimizer.

        Returns
        -------
        None
        """
        if not isinstance(loss_function, Loss):
            raise ValueError("`loss_function` must be of type `Loss`.")
        if not isinstance(optimizer, Optimizer):
            raise ValueError("`optimizer` must be of type `Optimizer`.")

        self.__is_compiled__ = True
        self.optimizer = optimizer
        self.loss_function = loss_function

    def fit(self, epochs=1000, batch_size=None, verbose=1, callbacks=[], get_output=False):
        """
        Trains the model using the provided Loss function, optimizer, and other parameters.

        This method performs the training process for the neural network. It includes
        forward and backward passes, loss and accuracy computation and parameter updates
        using the chosen optimizer. During training, it also keeps track of the loss and
        accuracy for both the training and validation datasets (if validation data is provided).

        Parameters
        ----------
        epochs : int, optional
            The number of epochs for training. The default is 1000.
        batch_size : int, optional
            The number of samples per batch. If None, the entire dataset is used for each pass.
        verbose : int, optional
            The verbosity level for printing metrics during training. Default is 1.
        callbacks : list of Callback, optional
            A list of callback functions to extend training functionality. Each callback should be a callable
            object that implements the following methods:
            - `on_train_start(model)`: Called at the start of the training loop.
            - `on_epoch_start(model)`: Called at the start of each epoch.
            - `on_epoch_end(model)`: Called at the end of each epoch. If the callback is an instance
              of `LiveMetrics`, a `figure` parameter should be passed.
            - `on_train_end(model)`: Called at the end of the training loop.
            The default is an empty list.
        get_output : bool, optional
            If True the final output is computed. Default is True.

        Raises
        ------
        ValueError
            - If the model has not been compiled before fitting.
            - If `batch_size` is not of type `int` and is not `None`.
            - If `epochs` is not of type `int`.
            - If `verbose` is not of type `int` and is not `None`.
        TypeError
            - If `callbacks` is not a list of Callback objects.

        Returns
        -------
        None
        """
        if not self.__is_compiled__:
            raise ValueError("The model must be compiled before fitting.")
        if not isinstance(batch_size, int) and batch_size is not None:
            raise ValueError("`batch_size` must be of type `int` or `None`.")
        if not isinstance(epochs, int):
            raise ValueError("`epochs` must be of type `int`.")
        if not isinstance(verbose, int) and verbose is not None:
            raise ValueError("`verbose` must be of type `int` or `None`.")
        if not isinstance(callbacks, list) or not all(isinstance(callback, Callback) for callback in callbacks):
            raise TypeError("`callbacks` must be a list of Callback objects.")

        self.__is_trained__ = True

        start_time = time.time()

        if self.validation:
            self.val_layers = deepcopy(self.train_layers)

        self.batch_size = batch_size

        if self.batch_size is None:
            steps = 1
        else:
            steps = self.train_inputs.shape[0] // self.batch_size
            if steps * self.batch_size < self.train_inputs.shape[0]:
                steps += 1
            if self.validation:
                self.val_batch_size = self.val_inputs.shape[0] // steps
                if self.val_batch_size <= 0:
                    raise ValueError(
                        f"Validation batch size must be at least 1. (currently {self.val_batch_size}). "
                        f"Ensure validation input size is greater than {steps} or increase training batch size to "
                        f"{self.train_inputs.shape[0] // self.val_inputs.shape[0]}."
                    )

        if epochs > 1000:
            update = 25
        elif 1000 >= epochs > 100:
            update = 10
        elif epochs <= 100:
            update = 1

        loss = 0.0
        acc = 0.0

        tqdm_epochs = False
        tqdm_steps = False

        if verbose == 1:
            tqdm_epochs = True
        elif verbose is not None and verbose != 0 and verbose != 1:
            tqdm_steps = True

        for callback in callbacks:
            if isinstance(callback, LiveMetrics):
                figure = plt.figure()
            callback.on_train_start(self)

        for epoch in (epoch_bar := tqdm(range(epochs), disable=not tqdm_epochs, file=sys.stdout)):
            epoch_bar.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            epoch_bar.set_postfix({"loss": loss, "accuracy": acc})

            for callback in callbacks:
                callback.on_epoch_start(self)

            train_accumulated_loss = 0
            train_accumulated_accuracy = 0

            val_accumulated_loss = 0
            val_accumulated_accuracy = 0

            for step in (step_bar := tqdm(range(steps), disable=not tqdm_steps, file=sys.stdout)):
                step_bar.set_description(f"Epoch [{epoch + 1}/{epochs}]")
                step_bar.set_postfix({"loss": loss, "accuracy": acc})

                for callback in callbacks:
                    callback.on_step_start(self)

                if self.batch_size is None:
                    self.train_input_batch = self.train_inputs
                    self.train_target_batch = self.train_targets

                    if self.validation:
                        self.val_input_batch = self.val_inputs
                        self.val_target_batch = self.val_targets
                else:
                    self.train_input_batch = Tensor(self.train_inputs.array[step * self.batch_size:(step + 1) * self.batch_size], requires_grad=True)
                    self.train_target_batch = Tensor(self.train_targets.array[step * self.batch_size:(step + 1) * self.batch_size], requires_grad=True)

                    if self.validation:
                        self.val_input_batch = Tensor(self.val_inputs.array[
                                               step * self.val_batch_size:(step + 1) * self.val_batch_size], requires_grad=True)
                        self.val_target_batch = Tensor(self.val_targets.array[
                                                step * self.val_batch_size:(step + 1) * self.val_batch_size], requires_grad=True)

                self.forward()

                for layer in self.train_layers:
                    layer.zero_grad()

                self.loss_function.zero_grad()

                self.backward()

                for i in range(len(self.train_layers)):
                    self.train_layers[i] = self.optimizer.update_layer(self.train_layers[i])

                    if self.validation:
                        if isinstance(self.val_layers[i], Dense) or isinstance(self.val_layers[i], Convolution2D):
                            self.val_layers[i].weights = self.train_layers[i].weights
                            self.val_layers[i].biases = self.train_layers[i].biases

                self.optimizer.step += 1

                loss = np.around(self.loss_function.output.array, 5)
                acc = accuracy(self.train_target_batch, self.train_output_batch)

                train_accumulated_loss += loss
                train_accumulated_accuracy += acc

                if self.validation:
                    val_accumulated_loss += self.loss_function.forward(self.val_target_batch, self.val_output_batch)
                    val_accumulated_accuracy += accuracy(self.val_target_batch, self.val_output_batch)

                for callback in callbacks:
                    callback.on_step_end(self)

            self.train_loss = train_accumulated_loss / steps
            self.train_accuracy = train_accumulated_accuracy / steps

            if epoch % update == 0 or epoch == 1:
                self.train_loss_history.append(self.train_loss)
                self.train_accuracy_history.append(self.train_accuracy)

            if self.validation:
                self.val_loss = (val_accumulated_loss / steps).array
                self.val_accuracy = val_accumulated_accuracy / steps

                if epoch % update == 0 or epoch == 1:
                    self.val_loss_history.append(self.val_loss)
                    self.val_accuracy_history.append(self.val_accuracy)

            self.verbose(verbose, epoch, epochs, start_time)

            for callback in callbacks:
                if isinstance(callback, LiveMetrics):
                    if epoch % update == 0 or epoch == 1:
                        callback.on_epoch_end(self, figure)
                else:
                    callback.on_epoch_end(self)

        if get_output:
            if self.batch_size is not None:
                self.train_outputs = Tensor(self.predict(self.train_inputs.array[:self.batch_size]), requires_grad=True)
                for step in range(1, steps):
                    self.train_outputs = Tensor(np.concatenate((self.train_outputs.array, Tensor(self.predict(
                        self.train_inputs.array[step * self.batch_size:(step + 1) * self.batch_size]), requires_grad=True)), axis=0))
            else:
                self.train_outputs = Tensor(self.predict(self.train_inputs), requires_grad=True)

        for callback in callbacks:
            callback.on_train_end(self)

    def results(self):
        """
        Plots the loss and accuracy evolution over epochs.

        This method generates plots to visualize the progress of the training
        process, showing how the loss and accuracy change for both training
        and validation datasets (if validation data is provided). This helps
        assess whether the model is learning and if it is overfitting.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.

        Returns
        -------
        None
        """
        if self.__is_trained__:
            figure, axs = plt.subplots(1, 2)

            axs[0].set_title("Loss Evolution")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].plot(self.train_loss_history, label="training dataset")
            if self.validation:
                axs[0].plot(self.val_loss_history, label="validation dataset")

            axs[1].set_title("Accuracy Evolution")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Accuracy")
            axs[1].plot(self.train_accuracy_history, label="training dataset")
            if self.validation:
                axs[1].plot(self.val_accuracy_history, label="validation dataset")

            axs[0].legend()
            axs[1].legend()

            plt.show()
        else:
            raise RuntimeError("The model has not been trained yet.")

    def save_params(self, filename="parameters"):
        """
        Saves the trained model parameters (weights and biases) to an HDF5 file.

        This method saves the weights and biases of all layers into an HDF5 file. The filename
        is constructed by appending a timestamp to the specified base filename, followed by the .h5
        extension, ensuring unique filenames for each save.

        Parameters
        ----------
        filename : str, optional
            The base filename for the file. The actual filename will include a timestamp and .h5
            extension (e.g., "parameters_MM_DD_YYYY-HH_MM_SS.h5"). Default is "parameters".

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        TypeError
            If `filename` is not a string.

        Notes
        -----
        The saved file uses the HDF5 format (.h5) and includes parameters, momentums, and cache
        values for each layer. Timestamping prevents accidental overwrites of previous saves.
        """
        if not isinstance(filename, str):
            raise TypeError("`filename` must be a string.")

        if self.__is_trained__:
            date_time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")

            with h5py.File(filename + f"_{date_time}.h5", 'w') as f:
                for i in range(len(self.train_layers)):
                    f.create_group(f"layer{i}")
                    if hasattr(self.train_layers[i], 'parameters'):
                        for j in range(len(self.train_layers[i].parameters)):
                            f[f"layer{i}"].create_group(f"parameter{j}")
                            f[f"layer{i}/parameter{j}"].create_dataset(f"value", data=
                            self.train_layers[i].parameters[j].array)
                            f[f"layer{i}/parameter{j}"].create_dataset(f"momentums", data=
                            self.train_layers[i].parameters[j].momentums.array)
                            f[f"layer{i}/parameter{j}"].create_dataset(f"cache", data=
                            self.train_layers[i].parameters[j].cache.array)

                    elif isinstance(self.train_layers[i], LSTM):
                        for j in range(len(self.train_layers[i].cells)):
                            f[f"layer{i}"].create_group(f"cell{j}")
                            for k in range(len(self.train_layers[i].cells[j].parameters)):
                                f[f"layer{i}/cell{j}"].create_group(f"parameter{k}")
                                f[f"layer{i}/cell{j}/parameter{k}"].create_dataset(f"value", data=
                                self.train_layers[i].cells[j].parameters[k].array)
                                f[f"layer{i}/cell{j}/parameter{k}"].create_dataset(f"momentums", data=
                                self.train_layers[i].cells[j].parameters[k].momentums.array)
                                f[f"layer{i}/cell{j}/parameter{k}"].create_dataset(f"cache", data=
                                self.train_layers[i].cells[j].parameters[k].cache.array)

            print(f"Data saved to {filename}")

        else:
            raise RuntimeError("The model has not been trained yet.")

    def load_params(self, path):
        """
        Loads the model parameters from a specified HDF5 file.

        This method restores parameters (weights, biases, momentums, and cache) from an HDF5
        file into the model's layers, enabling further training or evaluation without retraining.

        Parameters
        ----------
        path : str
            The file path to the HDF5 (.h5) file containing the saved parameters.

        Raises
        ------
        TypeError
            If `path` is not a string.
        ValueError
            If the file does not have a .h5 extension.

        Notes
        -----
        The file must be generated by the `save_params` method to ensure compatibility with the
        model's architecture and parameter structure.
        """
        if not isinstance(path, str):
            raise TypeError("`path` must be a string")

        _, extension = os.path.splitext(path)

        if extension != ".h5":
            raise ValueError("`extension` must be '.h5'")

        with h5py.File(path, 'r') as f:
            for i in range(len(self.train_layers)):
                if hasattr(self.train_layers[i], 'parameters'):
                    for j in range(len(self.train_layers[i].parameters)):
                        self.train_layers[i].parameters[j] = Parameter(
                            f[f"layer{i}/parameter{j}/value"].astype(np.float64)[:], requires_grad=True)
                        self.train_layers[i].parameters[j].momentums = Tensor(
                            f[f"layer{i}/parameter{j}/momentums"].astype(np.float64)[:])
                        self.train_layers[i].parameters[j].cache = Tensor(
                            f[f"layer{i}/parameter{j}/cache"].astype(np.float64)[:])

                elif isinstance(self.train_layers[i], LSTM):
                    for j in range(len(self.train_layers[i].cells)):
                        for k in range(len(self.train_layers[i].cells[j].parameters)):
                            self.train_layers[i].cells[j].parameters[k] = Parameter(
                                f[f"layer{i}/cell{j}/parameter{k}/value"].astype(np.float64)[:], requires_grad=True)
                            self.train_layers[i].cells[j].parameters[k].momentums = Tensor(
                                f[f"layer{i}/cell{j}/parameter{k}/momentums"].astype(np.float64)[:])
                            self.train_layers[i].cells[j].parameters[k].cache = Tensor(
                                f[f"layer{i}/cell{j}/parameter{k}/cache"].astype(np.float64)[:])

        print(f"Data loaded from {path}")

    def save_histories(self, filename="metrics_history", extension="h5"):
        """
        Saves the training and validation histories to a file.

        This method saves the training and validation histories (loss and accuracy) to a file with the specified filename and extension.
        The histories are saved with a timestamp appended to the base filename.

        Parameters
        ----------
        filename : str, optional
            The base filename of the file to save the histories. The default is "metrics_history".
        extension : str, optional
            The file extension to use for saving the histories. The default is "h5".

        Returns
        -------
        histories : dict
            A dictionary containing the training and validation histories. The dictionary includes the following keys:
            - "train_loss": List of training loss values.
            - "train_accuracy": List of training accuracy values.
            - "val_loss": List of validation loss values (if validation is enabled).
            - "val_accuracy": List of validation accuracy values (if validation is enabled).

        Raises
        ------
        TypeError
            If `filename` is not a string.
            If `extension` is not a string.
        ValueError
            If `extension` is not either 'h5' or 'pkl'.

        Notes
        -----
        The histories are saved in a file with a timestamp appended to the base filename. The file format can be either HDF5 (.h5) or pickle (.pkl).
        """
        if not isinstance(filename, str):
            raise TypeError("`filename` must be a string")
        if not isinstance(extension, str):
            raise TypeError("`extension` must be a string")
        if extension not in ("h5", "pkl"):
            raise ValueError("`extension` must be either 'h5' or 'pkl'")

        date_time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        histories = {
            "train_loss": self.train_loss_history,
            "train_accuracy": self.train_accuracy_history
        }
        if self.validation:
            histories["val_loss"] = self.val_loss_history
            histories["val_accuracy"] = self.val_accuracy_history

        if extension == "pkl":
            with open(filename + f"_{date_time}.pkl", 'wb') as f:
                pickle.dump(histories, f)
        elif extension == "h5":
            with h5py.File(filename + f"_{date_time}.h5", 'w') as f:
                f.create_dataset(f"train_loss", data=histories["train_loss"])
                f.create_dataset(f"train_accuracy", data=histories["train_accuracy"])
                if self.validation:
                    f.create_dataset(f"val_loss", data=histories["val_loss"])
                    f.create_dataset(f"val_accuracy", data=histories["val_accuracy"])

        print(f"Data saved to {filename}")

    def summary(self):
        """
        Prints a summary of the model architecture.

        This method prints the shape of the output for each layer in the model.

        Returns
        -------
        None
        """
        params_count = 0

        self.predict(self.train_inputs.array[0].reshape(1, *self.train_inputs.array[0].shape))
        for i in range(len(self.train_layers)):

            if hasattr(self.train_layers[i], 'parameters'):
                for param in self.train_layers[i].parameters:
                    params_count += param.size

            elif isinstance(self.train_layers[i], LSTM):
                for cell in self.train_layers[i].cells:
                    for param in cell.parameters:
                        params_count += param.size

            print(f"{type(self.train_layers[i]).__name__}: {self.train_layers[i].outputs.shape}")
        print(f"\nNumber of parameters: {params_count}\n")