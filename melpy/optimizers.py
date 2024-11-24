from .functions import *

class Optimizer:
    def __init__(self):
        self.learning_rate = None
        self.momentum = None
    
    def update_params(self, layer):
        pass
    
class SGD():
    def __init__(self):
        super().__init__()
    
    def update_params(self, layer):
        if isinstance(layer, Dense):
            if self.momentum is not None:
                weights = self.momentum * layer.w_momentum - layer.dW * self.learning_rate
                biases = self.momentum * layer.b_momentum - layer.dB * self.learning_rate
                
                layer.w_momentum = weights
                layer.b_momentum = biases
                layer.weights += weights
                layer.biases += biases
                
            elif self.momentum is None:
                layer.weights -= layer.dW * self.learning_rate
                layer.biases -= layer.dB * self.learning_rate
                
        elif isinstance(layer, Convolution2D):
            if self.momentum is not None:
                filters = self.momentum * layer.w_momentum - layer.dW * self.learning_rate
                layer.w_momentum = filters
                layer.filters += filters
                             
            elif self.momentum is None:
                layer.filters -= layer.dW * self.learning_rate
         
        return layer