import numpy as np
from .tensor import *
from .layers import *

class Loss:
    """
    Base class for all loss functions.

    Methods
    -------
    forward(targets : Tensor, predictions : Tensor)
        Computes the loss.
    backward(targets : Tensor, predictions : Tensor)
        Computes the derivative of the loss.
    """
    def __init__(self):
        """
        Initializes the Loss class.
        """
        self.targets = None
        self.predictions = None
        self.output = None
        pass

    def forward(self, targets, predictions):
        """
        Computes the loss.

        Parameters
        ----------
        targets : Tensor
            Target data.
        predictions : Tensor
            Output data.

        Returns
        -------
        float
            The computed loss.
        """
        pass

    def backward(self):
        """
        Computes the derivative of the loss.

        Returns
        -------
        Tensor
            The derivative of the loss.
        """
        pass

    def zero_grad(self):
        pass

class MSE(Loss):
    """
    A class to perform the mean squared error loss.

    Methods
    -------
    forward(targets : Tensor, predictions : Tensor)
        Computes the mean squared error.
    """
    def __init__(self):
        """
        Initializes the MSE class.
        """
        super().__init__()

    def forward(self, targets, predictions):
        """
        Computes the mean squared error.

        Parameters
        ----------
        targets : Tensor
            Target data.
        predictions : Tensor
            Output data.

        Returns
        -------
        float
            The mean squared error.
        """
        self.targets = targets
        self.predictions = predictions

        diff = self.targets - self.predictions
        self.output = sum(diff * diff) / diff.size
        return self.output

    def backward(self):
        self.output.backward(1)
        return self.predictions.grad

    def zero_grad(self):
        self.output.zero_grad()

class BinaryCrossEntropy(Loss):
    """
    A class to perform binary cross entropy loss.

    Methods
    -------
    forward(targets : Tensor, predictions : Tensor)
        Computes the binary cross entropy loss.
    backward(targets : Tensor, predictions : Tensor)
        Computes the binary cross entropy derivative.
    """
    def __init__(self):
        """
        Initializes the BinaryCrossEntropy class.
        """
        super().__init__()

    def forward(self, targets, predictions):
        """
        Computes the binary cross entropy loss.

        Parameters
        ----------
        targets : Tensor
            Target data.
        predictions : Tensor
            Output data.

        Returns
        -------
        float
            The binary cross entropy loss.
        """
        self.targets = targets
        self.predictions = predictions

        e = 1e-10
        self.output = -(sum(self.targets * log(self.predictions + e) +
                        (1-self.targets) * log(1-self.predictions + e))) / len(self.targets)
        return self.output

    def backward(self):
        """
        Computes the binary cross entropy derivative.

        Returns
        -------
        Tensor
            The binary cross entropy derivative.
        """
        self.output.backward(1)
        return self.predictions.grad

    def zero_grad(self):
        self.output.zero_grad()


class CategoricalCrossEntropy(Loss):
    """
    A class to compute categorical cross-entropy loss and its derivative.

    Methods
    -------
    forward(targets : Tensor, predictions : Tensor)
        Computes the categorical cross-entropy loss.
    backward(targets : Tensor, predictions : Tensor)
        Computes the categorical cross-entropy derivative.
    """

    def __init__(self):
        """
        Initializes the CategoricalCrossEntropy class.
        """
        super().__init__()

    def forward(self, targets, predictions, from_logits=False):
        """
        Computes the categorical cross-entropy loss.

        Parameters
        ----------
        targets : Tensor
            Target data. Can be one-hot encoded or integer labels.
        predictions : Tensor
            Output probabilities (e.g., softmax predictions).

        Returns
        -------
        float
            The categorical cross-entropy loss.
        """
        self.targets = targets
        self.predictions = predictions

        self.predictions = clip(self.predictions, a_min=1e-15, a_max=1 - 1e-15)

        if from_logits:
            softmax = Softmax()
            softmax.inputs = self.predictions
            softmax.forward()

            loss = -sum(self.targets * log(softmax.outputs), axis=1)
        else:
            loss = -sum(self.targets * log(self.predictions), axis=1)

        self.output = sum(loss)/loss.size
        return self.output

    def backward(self):
        """
        Computes the categorical cross-entropy derivative.

        Returns
        -------
        Tensor
            Gradient of the loss with respect to predictions.
        """
        self.output.backward(1)
        return self.predictions.grad

    def zero_grad(self):
        self.output.zero_grad()

class NegativeLogLikelihood(Loss):
    """
    A class to compute negative log-likelihood loss and its derivative.

    Methods
    -------
    forward(targets : Tensor, predictions : Tensor)
        Computes the negative log-likelihood loss.
    backward(targets : Tensor, predictions : Tensor)
        Computes the negative log-likelihood derivative.
    """

    def __init__(self):
        """
        Initializes the CategoricalCrossEntropy class.
        """
        super().__init__()

    def forward(self, targets, predictions):
        """
        Computes the negative log-likelihood loss.

        Parameters
        ----------
        targets : Tensor
            Target data. Can be one-hot encoded or integer labels.
        predictions : Tensor
            Output probabilities (e.g., softmax predictions).

        Returns
        -------
        float
            The negative log-likelihood loss.
        """
        raise NotImplementedError

    def backward(self):
        """
        Computes the categorical cross-entropy derivative.

        Returns
        -------
        Tensor
            Gradient of the loss with respect to predictions.
        """
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError