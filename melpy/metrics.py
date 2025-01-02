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
    if isinstance(targets, np.ndarray):
        targets = Tensor(targets)

    if isinstance(predictions, np.ndarray):
        predictions = Tensor(predictions)

    predicted_labels = np.argmax(predictions.array, axis=1)
    true_labels = np.argmax(targets.array, axis=1)

    return  np.sum(predicted_labels == true_labels) / len(true_labels)

def differences(targets, predictions):
    """
    Prints the differences between the target values and the predicted output values.

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
    if isinstance(targets, np.ndarray):
        targets = Tensor(targets)

    if isinstance(predictions, np.ndarray):
        predictions = Tensor(predictions)

    for i in range(len(predictions.array)):
        if predictions.array[i] >= targets.array[i]:
            print(predictions.array[i] - targets.array[i])
        if predictions.array[i] < targets.array[i]:
            print(targets.array[i] - predictions.array[i])