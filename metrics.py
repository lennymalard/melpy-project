import numpy as np

def accuracy(targets, outputs):
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
    for i in range(len(outputs)):
        if outputs[i] >= targets[i]:
            print(outputs[i] - targets[i])
        if outputs[i] < targets[i]:
            print(targets[i] - outputs[i])