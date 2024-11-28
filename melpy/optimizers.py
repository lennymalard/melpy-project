from .functions import *

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
                weights = self.momentum * layer.w_momentum - layer.dW * self.learning_rate
                if layer.biases is not None:
                    biases = self.momentum * layer.b_momentum - layer.dB * self.learning_rate

                layer.w_momentum = weights
                layer.b_momentum = biases
                layer.weights += weights
                layer.biases += biases

            elif self.momentum is None:
                layer.weights -= layer.dW * self.learning_rate
                if layer.biases is not None:
                    layer.biases -= layer.dB * self.learning_rate

        return layer

class Adam(Optimizer):
    def __init__(self):
        super().__init__()
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-7

    def update_params(self, layer):
        if isinstance(layer, Dense) or isinstance(layer, Convolution2D):
            weights = self.momentum * layer.w_momentum - layer.dW * self.learning_rate
            if layer.biases is not None:
                biases = self.momentum * layer.b_momentum - layer.dB * self.learning_rate

            layer.w_momentum = weights
            layer.b_momentum = biases
            layer.weights += weights
            layer.biases += biases

