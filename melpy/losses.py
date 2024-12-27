import numpy as np

class Loss:
    """
    Base class for all loss functions.

    Methods
    -------
    loss(targets : ndarray, predictions : ndarray)
        Computes the loss.
    derivative(targets : ndarray, predictions : ndarray)
        Computes the derivative of the loss.
    """
    def __init__(self):
        """
        Initializes the Loss class.
        """
        pass

    def loss(self, targets, predictions):
        """
        Computes the loss.

        Parameters
        ----------
        targets : ndarray
            Target data.
        predictions : ndarray
            Output data.

        Returns
        -------
        float
            The computed loss.
        """
        pass

    def derivative(self, targets, predictions):
        """
        Computes the derivative of the loss.

        Parameters
        ----------
        targets : ndarray
            Target data.
        predictions : ndarray
            Output data.

        Returns
        -------
        ndarray
            The derivative of the loss.
        """
        pass

class MSE(Loss):
    """
    A class to perform the mean squared error loss.

    Methods
    -------
    loss(targets : ndarray, predictions : ndarray)
        Computes the mean squared error.
    """
    def __init__(self):
        """
        Initializes the MSE class.
        """
        super().__init__()

    def loss(self, targets, predictions):
        """
        Computes the mean squared error.

        Parameters
        ----------
        targets : ndarray
            Target data.
        predictions : ndarray
            Output data.

        Returns
        -------
        float
            The mean squared error.
        """
        diff = targets - predictions
        return np.sum(diff * diff) / np.size(diff)

class BinaryCrossEntropy(Loss):
    """
    A class to perform binary cross entropy loss.

    Methods
    -------
    loss(targets : ndarray, predictions : ndarray)
        Computes the binary cross entropy loss.
    derivative(targets : ndarray, predictions : ndarray)
        Computes the binary cross entropy derivative.
    """
    def __init__(self):
        """
        Initializes the BinaryCrossEntropy class.
        """
        super().__init__()

    def loss(self, targets, predictions):
        """
        Computes the binary cross entropy loss.

        Parameters
        ----------
        targets : ndarray
            Target data.
        predictions : ndarray
            Output data.

        Returns
        -------
        float
            The binary cross entropy loss.
        """
        e = 1e-10
        return -(np.sum(targets * np.log(predictions + e) +
                        (1-targets) * np.log(1-predictions + e))) / len(targets)

    def derivative(self, targets, predictions):
        """
        Computes the binary cross entropy derivative.

        Parameters
        ----------
        targets : ndarray
            Target data.
        predictions : ndarray
            Output data.

        Returns
        -------
        ndarray
            The binary cross entropy derivative.
        """
        e = 1e-10
        return -(targets / predictions - (1 - targets + e) / (1 - predictions + e)) / len(predictions)


class CategoricalCrossEntropy(Loss):
    """
    A class to compute categorical cross-entropy loss and its derivative.

    Methods
    -------
    loss(targets : ndarray, predictions : ndarray)
        Computes the categorical cross-entropy loss.
    derivative(targets : ndarray, predictions : ndarray)
        Computes the categorical cross-entropy derivative.
    """

    def __init__(self):
        """
        Initializes the CategoricalCrossEntropy class.
        """
        super().__init__()

    def loss(self, targets, predictions):
        """
        Computes the categorical cross-entropy loss.

        Parameters
        ----------
        targets : ndarray
            Target data. Can be one-hot encoded or integer labels.
        predictions : ndarray
            Output probabilities (e.g., softmax predictions).

        Returns
        -------
        float
            The categorical cross-entropy loss.
        """
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)

        if len(targets.shape) == 1:
            correct_confidences = predictions_clipped[
                range(len(predictions)), targets
            ]
        elif len(targets.shape) == 2:
            correct_confidences = np.sum(predictions_clipped * targets, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return np.mean(negative_log_likelihoods)

    def derivative(self, targets, predictions):
        """
        Computes the categorical cross-entropy derivative.

        Parameters
        ----------
        targets : ndarray
            Target data. Can be one-hot encoded or integer labels.
        predictions : ndarray
            Output probabilities (e.g., softmax predictions).

        Returns
        -------
        ndarray
            Gradient of the loss with respect to predictions.
        """
        if len(targets.shape) == 1:
            targets = np.eye(predictions.shape[1])[targets]

        return (predictions - targets) / len(predictions)