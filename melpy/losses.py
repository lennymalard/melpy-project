import numpy as np

class Loss:
    """
    Base class for all loss functions.

    Methods
    -------
    loss(targets : ndarray, outputs : ndarray)
        Computes the loss.
    derivative(targets : ndarray, outputs : ndarray)
        Computes the derivative of the loss.
    """
    def __init__(self):
        """
        Initializes the Loss class.
        """
        pass

    def loss(self, targets, outputs):
        """
        Computes the loss.

        Parameters
        ----------
        targets : ndarray
            Target data.
        outputs : ndarray
            Output data.

        Returns
        -------
        float
            The computed loss.
        """
        pass

    def derivative(self, targets, outputs):
        """
        Computes the derivative of the loss.

        Parameters
        ----------
        targets : ndarray
            Target data.
        outputs : ndarray
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
    loss(targets : ndarray, outputs : ndarray)
        Computes the mean squared error.
    """
    def __init__(self):
        """
        Initializes the MSE class.
        """
        super().__init__()

    def loss(self, targets, outputs):
        """
        Computes the mean squared error.

        Parameters
        ----------
        targets : ndarray
            Target data.
        outputs : ndarray
            Output data.

        Returns
        -------
        float
            The mean squared error.
        """
        diff = targets - outputs
        return np.sum(diff * diff) / np.size(diff)

class BinaryCrossEntropy(Loss):
    """
    A class to perform binary cross entropy loss.

    Methods
    -------
    loss(targets : ndarray, outputs : ndarray)
        Computes the binary cross entropy loss.
    derivative(targets : ndarray, outputs : ndarray)
        Computes the binary cross entropy derivative.
    """
    def __init__(self):
        """
        Initializes the BinaryCrossEntropy class.
        """
        super().__init__()

    def loss(self, targets, outputs):
        """
        Computes the binary cross entropy loss.

        Parameters
        ----------
        targets : ndarray
            Target data.
        outputs : ndarray
            Output data.

        Returns
        -------
        float
            The binary cross entropy loss.
        """
        e = 1e-10
        return -(np.sum(targets * np.log(outputs + e) +
                        (1-targets) * np.log(1-outputs + e))) / len(targets)

    def derivative(self, targets, outputs):
        """
        Computes the binary cross entropy derivative.

        Parameters
        ----------
        targets : ndarray
            Target data.
        outputs : ndarray
            Output data.

        Returns
        -------
        ndarray
            The binary cross entropy derivative.
        """
        e = 1e-10
        return -(targets / outputs - (1 - targets + e) / (1 - outputs + e)) / len(outputs)

class CategoricalCrossEntropy(Loss):
    """
    A class to perform categorical cross entropy loss.

    Methods
    -------
    loss(targets : ndarray, outputs : ndarray)
        Computes the categorical cross entropy loss.
    derivative(targets : ndarray, outputs : ndarray)
        Computes the categorical cross entropy derivative.
    """
    def __init__(self):
        """
        Initializes the CategoricalCrossEntropy class.
        """
        super().__init__()

    def loss(self, targets, outputs):
        """
        Computes the categorical cross entropy loss.

        Parameters
        ----------
        targets : ndarray
            Target data.
        outputs : ndarray
            Output data.

        Returns
        -------
        float
            The categorical cross entropy loss.
        """
        targets_clipped = np.clip(outputs, 1e-7, 1 - 1e-7)
        if len(targets.shape) == 1:
            correct_confidences = targets_clipped[
            range(len(outputs)),
            targets
            ]
        elif len(targets.shape) == 2:
            correct_confidences = np.sum(
            targets_clipped*targets,
            axis=1
            )
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

    def derivative(self, targets, outputs):
        """
        Computes the categorical cross entropy derivative.

        Parameters
        ----------
        targets : ndarray
            Target data.
        outputs : ndarray
            Output data.

        Returns
        -------
        ndarray
            The categorical cross entropy derivative.
        """
        if len(targets.shape) == 1:
            targets = np.eye(len(outputs[0]))[targets]
        return ((-targets + 1e-5) / (outputs + 1e-5)) / len(outputs)