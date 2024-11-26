import numpy as np
from math import sqrt
from .im2col import *

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
        return -(np.sum(targets * np.log(outputs + e) + \
                        (1-targets) * np.log(1-outputs + e))) / len(targets)
    
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
    def __init__(self, n_in, n_out, weight_init="he_normal"):
        super().__init__()
        
        def initialize_weights(weight_init, n_in, n_out):
            if weight_init == "he_normal":
                weights = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            elif weight_init == "glorot_uniform":
                limit = np.sqrt(6 / (n_in + n_out))
                weights = np.random.uniform(-limit, limit, (n_in, n_out))
            elif weight_init == "he_uniform":
                limit = np.sqrt(6 / n_in)
                weights = np.random.uniform(-limit, limit, (n_in, n_out))
            elif weight_init == "glorot_normal":
                weights = np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))
            else:
                raise ValueError("invalid value for 'weight_init'")
            return weights
        
        self.weights = initialize_weights(weight_init, n_in, n_out)
        self.biases = np.random.rand(1,n_out)
        self.w_momentum = np.zeros_like(self.weights)
        self.b_momentum = np.zeros_like(self.biases)
        
    def forward(self):
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return self.outputs
    
    def backward(self, dX):
        self.dY = dX
        self.dW = np.dot(self.inputs.T, self.dY)
        self.dB = np.sum(self.dY, axis=0, keepdims=True)
        self.dX = np.dot(self.dY, self.weights.T)
 
        return self.dX
    
class ReLU(Layer):
    def __init__(self):
        super().__init__()
        
    def derivative(self):
        dA = np.ones_like(self.inputs)
        dA[self.inputs <= 0] = 0
        return dA
        
    def forward(self):
        self.outputs = np.maximum(0,self.inputs)
        return self.outputs
    
    def backward(self, dX):
        self.dY = dX
        self.dX = self.dY * self.derivative()
        return self.dX

class LeakyReLU(Layer):
    def __init__(self):
        super().__init__()
    
    def derivative(self):
        dA = np.ones_like(self.inputs)
        dA[self.inputs <= 0] = 0.01
        return dA
    
    def forward(self):
        self.outputs = np.where(self.inputs > 0, self.inputs, self.inputs * 0.01)
        return self.outputs
    
    def backward(self, dX):
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
        return self.outputs
    
    def backward(self, dX):
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
    
    def backward(self, dX):
        self.dY = dX
        self.dX = np.empty_like(dX)
        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, self.dX)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dX[index] = np.dot(jacobian_matrix,
            single_dvalues)
        return self.dX
    
class Convolution2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding="valid", stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = np.random.uniform(
            -sqrt(6 / (in_channels + out_channels)),
            sqrt(6 / (in_channels + out_channels)),
            (out_channels, in_channels, kernel_size, kernel_size)
        )
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.w_momentum = np.zeros_like(self.filters)
        
        if self.padding not in ["valid", "same"]:
            raise ValueError("invalid value for 'padding'")
        
        if self.padding == "same":
            self.stride = 1
        
    def calculate_padding(self):
         if self.padding == "valid":
             return (0, 0, 0, 0)
         elif self.padding == "same":
             input_height, input_width = self.inputs.shape[2], self.inputs.shape[3]
             if input_height % self.stride == 0:
                 pad_along_height = max((self.kernel_size - self.stride), 0)
             else:
                 pad_along_height = max(self.kernel_size - (input_height % self.stride), 0)
             if input_width % self.stride == 0:
                 pad_along_width = max((self.kernel_size - self.stride), 0)
             else:
                 pad_along_width = max(self.kernel_size - (input_width % self.stride), 0)
 
             pad_top = pad_along_height // 2
             pad_bottom = pad_along_height - pad_top
             pad_left = pad_along_width // 2
             pad_right = pad_along_width - pad_left
 
             return (pad_top, pad_bottom, pad_left, pad_right)
 
    def explicit_padding(self):
        pad_top, pad_bottom, pad_left, pad_right = self.calculate_padding()
        return np.pad(self.inputs, ((0, 0), (0, 0), (pad_top, pad_bottom), \
                                    (pad_left, pad_right)), mode='constant')

    def get_output_size(self, input_heighteight, input_widthidth):
        if self.padding == 'valid':
            output_height = (input_heighteight - self.kernel_size) // self.stride + 1
            output_width = (input_widthidth - self.kernel_size) // self.stride + 1
        elif self.padding == 'same':
            output_height = np.ceil(input_heighteight / self.stride)
            output_width = np.ceil(input_widthidth / self.stride)
            
        return int(output_height), int(output_width)

    def forward(self):
        self.input_padded = self.explicit_padding()

        self.input_cols = im2col(self.input_padded, self.kernel_size, self.stride)
        self.filter_cols = self.filters.reshape(self.out_channels, -1)

        output_height, output_width = self.get_output_size(self.inputs.shape[2], self.inputs.shape[3])

        self.output_cols = self.filter_cols @ self.input_cols

        self.outputs = np.array(np.hsplit(self.output_cols, self.inputs.shape[0])).reshape(
            (self.input_padded.shape[0], self.out_channels, output_height, output_width)
        )

        return self.outputs

    def backward(self, dX):
        self.dY = dX
    
        self.dY_reshaped = self.dY.reshape(self.dY.shape[0] * self.dY.shape[1], self.dY.shape[2] * self.dY.shape[3])
        self.dY_reshaped = np.array(np.vsplit(self.dY_reshaped, self.inputs.shape[0]))
        self.dY_reshaped = np.concatenate(self.dY_reshaped, axis=-1)
    
        self.dX_cols = self.filter_cols.T @ self.dY_reshaped
        self.dW_cols = self.dY_reshaped @ self.input_cols.T
    
        self.dX_padded = col2im(self.input_cols, self.input_padded.shape, self.kernel_size, self.stride)
    
        if self.padding == "same":
            (pad_top, pad_bottom, pad_left, pad_right) = self.calculate_padding()
            self.dX = self.dX_padded[:, :, pad_top:-pad_bottom, pad_left:-pad_right]
        else:
            self.dX = self.dX_padded
    
        self.dW = self.dW_cols.reshape((self.dW_cols.shape[0], self.in_channels, self.kernel_size, self.kernel_size))
    
        return self.dX

class Pooling2D(Layer):
    def __init__(self, pool_size, stride, mode="max"):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode
        
        if self.mode not in ["max"]: # Average sera à ajouter plus tard
            raise ValueError("invalid value for 'mode'")
        
    def derivative(self):
        pass
    
    def forward(self): 
        output_height = int((self.inputs.shape[2] - self.pool_size + self.stride)//self.stride)
        output_width = int((self.inputs.shape[3] - self.pool_size + self.stride)//self.stride)
        
        output_shape = (self.inputs.shape[0], self.inputs.shape[1], output_height, output_width)
        
        self.input_cols = im2col(self.inputs, self.pool_size, self.stride)
        self.input_cols_reshaped = np.array(np.hsplit(np.array(np.hsplit(self.input_cols, self.inputs.shape[0])), self.inputs.shape[1]))

        self.maxima = np.max(self.input_cols_reshaped, axis=2)
        self.maxima_reshaped = self.maxima.reshape(self.inputs.shape[1],-1)
        
        self.outputs = col2im(self.maxima_reshaped, output_shape, 1, 1) 
                     
        return self.outputs
    
    def backward(self, dX):
        self.dY = dX
        self.dX = np.zeros_like(self.inputs)
        
        self.dY_cols = im2col(self.dY, 1, 1)
        self.dY_cols_reshaped = np.array(np.hsplit(np.array(np.hsplit(self.dY_cols, self.dY.shape[0])), self.dY.shape[1])).transpose(0,1,3,2)
        
        self.input_cols = im2col(self.inputs, self.pool_size, self.stride)
        self.input_cols_reshaped = np.array(np.hsplit(np.array(np.hsplit(self.input_cols, self.inputs.shape[0])), self.inputs.shape[1])).transpose(0,1,3,2)
        
        self.output_cols = im2col(self.outputs, 1, 1)
        self.output_cols_reshaped = np.array(np.hsplit(np.array(np.hsplit(self.output_cols, self.inputs.shape[0])), self.inputs.shape[1])).transpose(0,1,3,2)
        
        self.mask = np.array(self.input_cols_reshaped == self.output_cols_reshaped, dtype = np.uint64)
        
        self.dX_cols = np.concatenate(np.concatenate(np.array(self.mask*self.dY_cols_reshaped).transpose(0,1,3,2), axis=1), axis=1)
        self.dX = col2im(self.dX_cols, self.inputs.shape, self.pool_size, self.stride)
        
        return self.dX
           
class Flatten(Layer):
    def __init__(self):
        super().__init__()
        
    def derivative(self):
        pass
    
    def forward(self): 
        self.outputs = self.inputs.reshape((self.inputs.shape[0], -1))
        return self.outputs

    def backward(self, dX):
        self.dY = dX
        self.dX = self.dY.reshape(self.inputs.shape)
        return self.dX

class Dropout(Layer):
    def __init__(self, p):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1")
        self.p = p
        self.mask = None
        self.training = True
        
    def derivative(self):
        pass
    
    def forward(self): 
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, size=self.inputs.shape)
            self.outputs =  self.inputs * self.mask * 1.0 / (1.0 - self.p)
        else:
            self.outputs = self.inputs
        return self.outputs

    def backward(self, dX):
        if self.training:
            self.dX = dX * self.mask * 1.0 / (1.0 - self.p)
        else:
            self.dX = dX
        return self.dX

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