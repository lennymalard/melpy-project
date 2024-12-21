import numpy as np

def accuracy(targets, predictions):
    """
    Computes the accuracy of the model's predictions.

    Parameters
    ----------
    targets : ndarray
        The true target values.
    predictions : ndarray
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
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(targets, axis=1)

    return  np.sum(predicted_labels == true_labels) / len(true_labels)

def differences(targets, predictions):
    """
    Prints the differences between the target values and the predicted output values.

    Parameters
    ----------
    targets : ndarray
        The true target values.
    predictions : ndarray
        The predicted output values.

    Raises
    ------
    ValueError
        If the lengths of `targets` and `outputs` are not the same.
    """
    for i in range(len(outputs)):
        if outputs[i] >= targets[i]:
            print(outputs[i] - targets[i])
        if outputs[i] < targets[i]:
            print(targets[i] - outputs[i])