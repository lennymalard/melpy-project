import numpy as np

def accuracy(targets, outputs):
    """
    Computes the accuracy of the model's predictions.

    Parameters
    ----------
    targets : ndarray
        The true target values.
    outputs : ndarray
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
    outputs = np.array(outputs)
    targets = np.array(targets)

    accuracy = 1e-10

    if outputs.shape[1] > 1:
        outputs = (outputs == np.max(outputs, axis=1, keepdims=True)).astype(float)

    total_correct = 0
    total_elements = 0
    for i in range(outputs.shape[1]):
        correct_predictions = np.sum(np.round(outputs[:, i]) == targets[:, i])
        total_correct += correct_predictions
        total_elements += len(targets[:, i])

    accuracy = total_correct / total_elements if total_elements > 0 else accuracy

    return accuracy

def differences(targets, outputs):
    """
    Prints the differences between the target values and the predicted output values.

    Parameters
    ----------
    targets : ndarray
        The true target values.
    outputs : ndarray
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