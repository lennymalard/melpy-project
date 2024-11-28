from .functions import *
from math import sqrt

class Optimizer:
    """
    Base class for optimizers used to update model parameters during training.

    Attributes
    ----------
    learning_rate : float
        The learning rate for the optimizer.
    momentum : float
        The momentum term for the optimizer.

    Methods
    -------
    update_params(layer : melpy.Layer)
        Updates the parameters of the given layer.
    """
    def __init__(self):
        """
        Initializes the Optimizer object.
        """
        self.learning_rate = None
        self.momentum = None

    def update_params(self, layer):
        """
        Updates the parameters of the given layer.

        Parameters
        ----------
        layer : melpy.Layer
            The layer whose parameters are to be updated.
        """
        pass

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Methods
    -------
    update_params(layer : melpy.Layer)
        Updates the parameters of the given layer using SGD.
    """
    def __init__(self):
        """
        Initializes the SGD optimizer.
        """
        super().__init__()

    def update_params(self, layer):
        """
        Updates the parameters of the given layer using SGD.

        Parameters
        ----------
        layer : melpy.Layer
            The layer whose parameters are to be updated.

        Returns
        -------
        layer : melpy.Layer
            The layer with updated parameters.
        """
        if isinstance(layer, Dense) or isinstance(layer, Convolution2D):
            if self.momentum is not None:
                weights = self.momentum * layer.weight_momentums - layer.dW * self.learning_rate
                if layer.biases is not None:
                    biases = self.momentum * layer.bias_momentums - layer.dB * self.learning_rate

                layer.weight_momentums = weights
                layer.bias_momentums = biases
                layer.weights += weights
                layer.biases += biases

            elif self.momentum is None:
                layer.weights -= layer.dW * self.learning_rate
                if layer.biases is not None:
                    layer.biases -= layer.dB * self.learning_rate

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
    update_params(layer : melpy.Layer)
        Updates the parameters of the given layer using Adam.
    """
    def __init__(self):
        """
        Initializes the Adam optimizer.
        """
        super().__init__()
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-7
        self.step = 0

    def update_params(self, layer):
        """
        Updates the parameters of the given layer using Adam.

        Parameters
        ----------
        layer : melpy.Layer
            The layer whose parameters are to be updated.

        Returns
        -------
        layer : melpy.Layer
            The layer with updated parameters.
        """
        if isinstance(layer, Dense) or isinstance(layer, Convolution2D):
            layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dW
            layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dB

            weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1**self.step)
            bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1**self.step)

            layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dW**2
            layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dB**2

            weight_cache_corrected = layer.weight_cache / (1 - self.beta2**self.step)
            bias_cache_corrected = layer.bias_cache / (1 - self.beta2**self.step)

            weights = - self.learning_rate * weight_momentums_corrected / (sqrt(weight_cache_corrected) + self.epsilon)
            if layer.biases is not None:
                biases = - self.learning_rate * bias_momentums_corrected / (sqrt(bias_cache_corrected) + self.epsilon)

            layer.weights += weights
            layer.biases += biases

            self.step += 1

        return layer