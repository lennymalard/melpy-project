import numpy as np
from tqdm import tqdm
from .layers import Linear

class LinearRegression:
    """
    A class for performing linear regression.

    Attributes
    ----------
    linear : Linear
        The linear layer used for regression.
    inputs : ndarray
        The input data for training.
    targets : ndarray
        The target data for training.
    prediction : ndarray
        The predicted output.
    loss : float
        The computed loss.

    Methods
    -------
    forward()
        Performs the forward pass to compute the output.
    predict(X)
        Generates predictions for the input data `X`.
    backward(lr)
        Performs the backward pass to update the parameters.
    fit(epochs, lr)
        Trains the model for a specified number of epochs.
    """
    def __init__(self, train_inputs, train_targets):
        """
        Initializes the LinearRegression object with training data.

        Parameters
        ----------
        train_inputs : ndarray
            The input data for training.
        train_targets : ndarray
            The target data for training.
        """
        self.linear = Linear(train_inputs.shape[1], train_targets.shape[1])
        self.inputs = self.linear.inputs = train_inputs
        self.targets = self.linear.targets = train_targets
        self.prediction = None
        self.loss = None

        if not isinstance(train_inputs, np.ndarray) or not isinstance(train_targets, np.ndarray):
            raise TypeError("Input data must be of type ndarray.")

    def forward(self):
        """
        Performs the forward pass to compute the output.

        Returns
        -------
        outputs : ndarray
            The output data.
        """
        self.outputs = self.linear.forward()
        return self.outputs

    def predict(self, X):
        """
        Generates predictions for the input data `X`.

        Parameters
        ----------
        X : ndarray
            The input data for which predictions are to be made.

        Returns
        -------
        predictions : ndarray
            The predicted output for the input data `X`.
        """
        self.linear.inputs = X
        return self.linear.forward()

    def backward(self, lr):
        """
        Performs the backward pass to update the parameters.

        Parameters
        ----------
        lr : float
            The learning rate for updating the parameters.
        """
        self.linear.backward(lr)

    def fit(self, epochs, lr):
        """
        Trains the model for a specified number of epochs.

        Parameters
        ----------
        epochs : int
            The number of epochs for training.
        lr : float
            The learning rate for updating the parameters.
        """
        for epoch in tqdm(range(epochs)):
            self.forward()
            self.backward(lr)

