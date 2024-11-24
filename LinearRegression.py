import numpy as np
from tqdm import tqdm
from .functions import Linear

class LinearRegression():
    def __init__(self, train_inputs, train_targets):
        self.linear = Linear(train_inputs.shape[1], train_targets.shape[1])
        self.inputs = self.linear.inputs = train_inputs
        self.targets = self.linear.targets = train_targets
        self.prediction = None
        self.loss = None
        
    def forward(self): 
        self.outputs = self.linear.forward()
        return self.outputs
    
    def predict(self,X):
        self.linear.inputs = X
        return self.linear.forward()
      
    def backward(self,lr):
        self.linear.backward(lr)

        
    def fit(self, epochs, lr):
        for epoch in tqdm(range(epochs)):
            self.forward()
            self.backward(lr)
