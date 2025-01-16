from .layers import *
import numpy as np
from .tensor import *

class Optimizer:
    """
    Base class for optimizers used to update model parameters during training.

    Attributes
    ----------
    learning_rate : float
        The learning rate for the optimizer.
    gradnorm : float
        The threshold value for gradient normalization during optimization.

    Methods
    -------
    update_parameter(parameter : Parameter)
        Updates a given parameter.
    update_layer(layer : Layer)
        Updates the parameters of the given layer.
    """
    def __init__(self):
        """
        Initializes the Optimizer object.
        """
        self.learning_rate = None
        self.gradnorm = None
        self.step = 1

    def update_parameter(self, parameter):
        """
        Updates a given parameter.

        Parameters
        ----------
        parameter : Parameter
            The parameter updated.
        """
        pass

    def update_layer(self, layer):
        """
        Updates the parameters of the given layer.

        Parameters
        ----------
        layer : Layer
            The layer whose parameters are to be updated.
        """
        pass

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Methods
    -------
    update_layer(layer : Layer)
        Updates the parameters of the given layer using SGD.
    """
    def __init__(self, learning_rate=0.001, momentum=None, gradnorm=1e15):
        """
        Initializes the SGD optimizer.
        """
        if not isinstance(learning_rate, float):
            raise TypeError("`learning_rate` must be a float")
        if not isinstance(momentum, float) and momentum is not None:
            raise TypeError("`momentum` must be a float or None")

        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.gradnorm = gradnorm

    def update_parameter(self, parameter):
        """
        Updates a given parameter using SGD.

        Parameters
        ----------
        parameter : Parameter
            The parameter updated.
        """
        if np.linalg.norm(parameter.grad) >= self.gradnorm:
            parameter.grad = self.gradnorm * parameter.grad / np.linalg.norm(parameter.grad)
        if self.momentum is not None:
            update_value = Tensor(self.momentum * parameter.momentums.array - parameter.grad * self.learning_rate)
            parameter.momentums = update_value
            parameter += update_value
        else:
            parameter -= Tensor(parameter.grad * self.learning_rate)

        return parameter

    def update_layer(self, layer):
        """
        Updates the parameters of the given layer using SGD.

        Parameters
        ----------
        layer : Layer
            The layer whose parameters are to be updated.

        Returns
        -------
        layer : Layer
            The layer with updated parameters.
        """
        if not isinstance(layer, Layer):
            raise TypeError("'layer' must be of type 'Layer'.")

        if isinstance(layer, Dense) or isinstance(layer, Convolution2D):
            layer.weights = self.update_parameter(layer.weights)
            if layer.biases is not None:
                layer.biases = self.update_parameter(layer.biases)

        elif isinstance(layer, LSTM):
            for cell in layer.cells:
                cell.i_input_weights = self.update_parameter(cell.i_input_weights)
                cell.f_input_weights = self.update_parameter(cell.f_input_weights)
                cell.c_input_weights = self.update_parameter(cell.c_input_weights)
                cell.o_input_weights = self.update_parameter(cell.o_input_weights)

                cell.i_hidden_weights = self.update_parameter(cell.i_hidden_weights)
                cell.f_hidden_weights = self.update_parameter(cell.f_hidden_weights)
                cell.c_hidden_weights = self.update_parameter(cell.c_hidden_weights)
                cell.o_hidden_weights = self.update_parameter(cell.o_hidden_weights)

                if cell.biases is not None:
                    cell.i_biases = self.update_parameter(cell.i_biases)
                    cell.f_biases = self.update_parameter(cell.f_biases)
                    cell.c_biases = self.update_parameter(cell.c_biases)
                    cell.o_biases = self.update_parameter(cell.o_biases)

        return layer

class Adam(Optimizer):
    """
    Adam optimizer.

    Attributes
    ----------
    beta1 : float
        The exponential decay rate for the first moment estimates.
    beta2 : float
        The exponential decay rate for the second moment estimates.
    epsilon : float
        A small constant for numerical stability.
    step : int
        The current step of the optimizer.

    Methods
    -------
    update_layer(layer : Layer)
        Updates the parameters of the given layer using Adam.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, gradnorm=1e15):
        """
        Initializes the Adam optimizer.
        """
        if not isinstance(learning_rate, float):
            raise TypeError("`learning_rate` must be a float")
        if not isinstance(beta1, float):
            raise TypeError("`beta1` must be a float")
        if not isinstance(beta2, float):
            raise TypeError("`beta2` must be a float")

        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        self.epsilon = 1e-7
        self.step = 1
        self.gradnorm = gradnorm

    def update_parameter(self, parameter):
        """
        Updates a given parameter using Adam.

        Parameters
        ----------
        parameter : Parameter
            The parameter updated.
        """
        if np.linalg.norm(parameter.grad) >= self.gradnorm:
            parameter.grad = self.gradnorm * parameter.grad / np.linalg.norm(parameter.grad)

        parameter.momentums = Tensor(self.beta1 * parameter.momentums.array + (1 - self.beta1) * parameter.grad)
        momentums_corrected = Tensor(parameter.momentums.array / (1 - self.beta1 ** self.step))

        parameter.cache = Tensor(self.beta2 * parameter.cache.array + (1 - self.beta2) * parameter.grad ** 2)
        cache_corrected = Tensor(parameter.cache.array / (1 - self.beta2 ** self.step))

        update_value = Tensor(- self.learning_rate * momentums_corrected.array / (
                    np.sqrt(cache_corrected.array) + self.epsilon))

        parameter += update_value

        return parameter

    def update_layer(self, layer):
        """
        Updates the parameters of the given layer using Adam.

        Parameters
        ----------
        layer : Layer
            The layer whose parameters are to be updated.

        Returns
        -------
        layer : Layer
            The layer with updated parameters.
        """
        if not isinstance(layer, Layer):
            raise TypeError("'layer' must be of type 'Layer'.")
        if isinstance(layer, Dense) or isinstance(layer, Convolution2D):
            layer.weights = self.update_parameter(layer.weights)
            if layer.biases is not None:
                layer.biases = self.update_parameter(layer.biases)

        elif isinstance(layer, LSTM):
            for cell in layer.cells:
                cell.i_input_weights = self.update_parameter(cell.i_input_weights)
                cell.f_input_weights = self.update_parameter(cell.f_input_weights)
                cell.c_input_weights = self.update_parameter(cell.c_input_weights)
                cell.o_input_weights = self.update_parameter(cell.o_input_weights)

                cell.i_hidden_weights = self.update_parameter(cell.i_hidden_weights)
                cell.f_hidden_weights = self.update_parameter(cell.f_hidden_weights)
                cell.c_hidden_weights = self.update_parameter(cell.c_hidden_weights)
                cell.o_hidden_weights = self.update_parameter(cell.o_hidden_weights)

                if cell.biases is not None:
                    cell.i_biases = self.update_parameter(cell.i_biases)
                    cell.f_biases = self.update_parameter(cell.f_biases)
                    cell.c_biases = self.update_parameter(cell.c_biases)
                    cell.o_biases = self.update_parameter(cell.o_biases)

        return layer