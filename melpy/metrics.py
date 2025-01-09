import numpy as np
from .tensor import *

def accuracy(targets, predictions):
    """
    Computes the accuracy of the model's predictions.

    Parameters
    ----------
    targets : ndarray, Tensor
        The true target values.
    predictions : ndarray, Tensor
        The predicted output values.

    Returns
    -------
    float
        The accuracy of the model's predictions.

    Raises
    ------
    ValueError
        If the shapes of `targets` and `outputs` are incompatible.
    """
    if isinstance(targets, Tensor) or isinstance(targets, Operation):
        targets = targets.array

    if isinstance(predictions, Tensor) or isinstance(predictions, Operation):
        predictions = predictions.array

    if targets.shape[1] != 1 and predictions.shape[1] != 1:
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(targets, axis=1)
    else:
        predicted_labels = np.around(predictions)
        true_labels = np.around(targets)

    return  np.sum(predicted_labels == true_labels) / len(true_labels)

def mean_absolute_error(targets, predictions):
    """
    Prints the mean absolute error between the target values and the predicted output values.

    Parameters
    ----------
    targets : ndarray, Tensor
        The true target values.
    predictions : ndarray, Tensor
        The predicted output values.

    Raises
    ------
    ValueError
        If the lengths of `targets` and `outputs` are not the same.
    """
    if isinstance(targets, Tensor) or isinstance(targets, Operation):
        targets = targets.array

    if isinstance(predictions, Tensor) or isinstance(predictions, Operation):
        predictions = predictions.array

    return np.abs(targets - predictions).mean()