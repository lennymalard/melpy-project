import numpy as np

class Layer:
    def __init__(self):
        self.inputs = None
        self.targets = None
        self.outputs = None
        self.dX = None
        self.dY = None
        self.dW = None
        self.dB = None

    def derivative(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass

class MSE():   
    def __init__(self):
        super().__init__()
        
    def loss(self, targets, outputs):
        diff = targets - outputs
        return np.sum(diff * diff) / np.size(diff)
    
class Binary_CrossEntropy():   
    def __init__(self):
        super().__init__()
        
    def loss(self, targets, outputs):
        e = 1e-10
        return -(np.sum(targets * np.log(outputs + e) + (1-targets) * np.log(1-outputs + e))) / len(targets)
    
    def derivative(self, targets, outputs):
        e = 1e-10
        return -(targets / outputs - (1 - targets + e) / (1 - outputs + e)) / len(outputs) 
    
class Categorical_CrossEntropy():
    def __init__(self):
        super().__init__()
        
    def loss(self, targets, outputs):
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
        if len(targets.shape) == 1:
            targets = np.eye(len(outputs[0]))[targets]
        return ((-targets + 1e-5) / (outputs + 1e-5)) / len(outputs) 

class Dense(Layer):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weights = np.random.rand(n_in, n_out)
        self.biases = np.random.rand(1,n_out)
        self.w_momentum = np.zeros_like(self.weights)
        self.b_momentum = np.zeros_like(self.biases)
        
    def forward(self):
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return np.dot(self.inputs, self.weights) + self.biases
    
    def backward(self, dX, lr, momentum):
        self.dY = dX
        self.dW = np.dot(self.inputs.T, self.dY)
        self.dB = np.sum(self.dY, axis=0, keepdims=True)
        self.dX = np.dot(self.dY, self.weights.T)
        
        if momentum is not None:
            weights = momentum * self.w_momentum - self.dW * lr
            biases = momentum * self.b_momentum - self.dB * lr
            
            self.w_momentum = weights
            self.b_momentum = biases
            self.weights += weights
            self.biases += biases
            
        elif momentum is None:
            self.weights -= self.dW * lr
            self.biases -= self.dB * lr
        
        self.dW *= np.zeros(self.dW.shape, dtype = np.float64)
        self.dB *= np.zeros(self.dB.shape, dtype = np.float64)
        
        return self.dX
    
class ReLU(Layer):
    def __init__(self):
        super().__init__()
        
    def derivative(self):
        self.inputs[self.inputs<=0] = 0
        self.inputs[self.inputs>0] = 1
        return self.inputs
        
    def forward(self):
        self.outputs = np.maximum(0,self.inputs)
        return np.maximum(0,self.inputs)
    
    def backward(self, dX, lr, momentum):
        self.dY = dX
        self.dX = self.dY * self.derivative()
        return self.dX

class Leaky_ReLU(Layer):
    def __init__(self):
        super().__init__()
    
    def derivative(self):
        dA = np.ones_like(self.inputs)
        dA[self.inputs < 0] = 0.01
        return dA
    
    def forward(self):
        self.outputs = np.where(self.inputs > 0, self.inputs, self.inputs * 0.01)
        return np.where(self.inputs > 0, self.inputs, self.inputs * 0.01)
    
    def backward(self, dX, lr, momentum):
        self.dY = dX
        self.dX = self.dY * self.derivative()
        return self.dX
    
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
    
    def derivative(self):
        return self.outputs * (1-self.outputs)
    
    def forward(self):
        self.outputs = 1 / (1+np.exp(-self.inputs))
        return 1 / (1+np.exp(-self.inputs))
    
    def backward(self, dX, lr, momentum):
        self.dY = dX
        self.dX = self.dY * self.derivative()
        return self.dX

class Softmax(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self):
        exp_values = np.exp(self.inputs - np.max(self.inputs, axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
        self.outputs = probabilities
        return self.outputs
    
    def backward(self, dX, lr, momentum):
        dA = np.empty_like(dX)
        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, dX)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            dA[index] = np.dot(jacobian_matrix,
            single_dvalues)
        return dA

class Linear(Layer):
    def __init__(self, n_in, n_out):
        self.weights = np.random.rand(n_in, n_out)
        self.biases = np.random.rand(1,n_out)
    
    def forward(self):
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return np.dot(self.inputs, self.weights) + self.biases
    
    def backward(self, lr):
        self.dW = np.sum(-2*self.inputs*(self.targets-(self.inputs @ self.weights+self.biases))) / len(self.targets-self.outputs) 
        self.dB = np.sum(-2*(self.targets-(self.inputs @ self.weights+self.biases))) / len(self.targets-self.outputs) 
        self.weights -= self.dW * lr
        self.biases -= self.dB * lr 
        self.dW *= np.zeros(self.dW.shape, dtype = np.float64) 
        self.dB *= np.zeros(self.dB.shape, dtype = np.float64)