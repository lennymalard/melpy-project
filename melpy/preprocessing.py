class RobustScaler():
    def __init__(self):
        self.fit = False
        self.q1_list = []
        self.med_list = []
        self.q3_list = []
        
    def transform(self, X):
        import numpy as np
        self.X = np.copy(X)
        if self.fit == False:
            for i in range(len(self.X[0][:])):
               q1 = np.quantile(self.X[:,i], 0.25)
               self.q1_list.append(q1)
               median = np.quantile(self.X[:,i], 0.5)
               self.med_list.append(median)
               q3 = np.quantile(self.X[:,i], 0.75)
               self.q3_list.append(q3)
               for k in range(len(self.X[:][:])):
                   self.X[k,i] = (self.X[k,i] - median) / (q3 - q1)
            self.fit = True
            return self.X
        else:
            for i in range(len(self.X[0][:])):
               q1 = self.q1_list[i]
               median = self.med_list[i]
               q3 = self.q3_list[i]
               for k in range(len(self.X[:][:])):
                   self.X[k,i] = (self.X[k,i] - median) / (q3 - q1)
            return self.X

class MinMaxScaler():
    def __init__(self):
        self.fit = False
        self.min_list = []
        self.max_list = []
        
    def transform(self, X):
        import numpy as np
        self.X = np.copy(X)
        if self.fit == False:
            for i in range(len(self.X[0][:])):
               minimum = np.amin(self.X[:,i])
               self.min_list.append(minimum)
               maximum = np.amax(self.X[:,i])
               self.max_list.append(maximum)
               for k in range(len(self.X[:][:])):
                   self.X[k,i] = (self.X[k,i] - minimum) / (maximum - minimum)
            self.fit = True
            return self.X
        else:
            for i in range(len(self.X[0][:])):
               minimum = self.min_list[i]
               maximum =self.max_list[i]
               for k in range(len(self.X[:][:])):
                   self.X[k,i] = (self.X[k,i] - minimum) / (maximum - minimum)
            return self.X

class StandardScaler():
    def __init__(self):
        self.fit = False
        self.std_list = []
        self.mean_list = []
        
    def transform(self, X):
        import numpy as np
        self.X = np.copy(X)
        if self.fit == False:
            for i in range(len(self.X[0][:])):
               std = np.std(self.X[:,i])
               self.std_list.append(std)
               mean = np.mean(self.X[:,i])
               self.mean_list.append(mean)
               for k in range(len(self.X[:][:])):
                   self.X[k,i] = (self.X[k,i] - mean) / (std)
            self.fit = True
            return self.X
        else:
            for i in range(len(self.X[0][:])):
               std = self.std_list[i]
               mean = self.mean_list[i]
               for k in range(len(self.X[:][:])):
                   self.X[k,i] = (self.X[k,i] - mean) / (std)
            return self.X

class OneHotEncoder():
    def __init__(self):
        self.fit = False
        self.y = None
        self.L1 = []
        self.L2 = []
        self.L3 = []
        self.L4 = []
    
    def transform(self, y):
        import numpy as np
        self.y = np.copy(y)
        self.L4 = []
        if self.fit == False:
            self.L1 = []
            for e in y:
                if [e] not in self.L1:
                    self.L1.append([e])
            self.L2 = list(self.L1)
            for i in range(len(self.L2)):
                self.L2[i] = [0]*len(self.L2)
                self.L2[i][i] = 1
            self.L3 = np.concatenate((np.array(self.L2).reshape(-1,len(self.L2[0])), np.array(self.L1).reshape(-1,1)), axis=1)
            self.L4 = []
            self.fit = True
            
        for i in range(len(y)):
            for j in range(len(self.L3)):
                if y[i] == self.L3[j][-1]:
                    self.L4.append(list(self.L3[j][:-1]))
        
        return np.array(self.L4, dtype=np.intc)

class StringEncoder():
    def __init__(self):
        self.X = None
        self.y = None
        self.feature_indices = []
        self.encoders = []
        self.fit = False
        
    def transform(self, X):
        import numpy as np
        from melpy.preprocessing import OneHotEncoder
        self.X = np.copy(X)
  
        def is_string(X):
            for i in range(len(X[0,:])):
                if type(X[0,i]) == str:
                    return True
            return False
        
        if self.fit == False:
            while 1:
                for i in range(len(self.X[0,:])):
                    if type(self.X[0,i]) == str:
                        self.feature_indices.append(i)
                        self.encoders.append(OneHotEncoder())
                    
                        temp = self.encoders[-1].transform(self.X[:,self.feature_indices[-1]])
                        self.X = np.delete(self.X, self.feature_indices[-1], axis=1)
                  
                        for k in range(len(temp[0,:])):
                            self.X = np.insert(self.X, i+k, temp[:,k], axis=1)
                        break
                if is_string(self.X) == False:
                    break
            self.fit = True
        else:
            for i in range(len(self.feature_indices)):
                temp = self.encoders[i].transform(self.X[:,self.feature_indices[i]])
                self.X = np.delete(self.X, self.feature_indices[i], axis=1)
                
                for k in range(len(temp[0,:])):
                    self.X = np.insert(self.X, self.feature_indices[i]+k, temp[:,k], axis=1)
                    
        return np.array(self.X, dtype=np.float64)
                

class FeatureEncoder():
    def __init__(self):
        self.X = None
        self.y = None
        self.feature_indices = []
        self.encoders = []
        self.fit = False
        
    def fit_transform(self, X, index):
        import numpy as np
        from melpy.preprocessing import OneHotEncoder
        self.X = np.copy(X)

        self.feature_indices.append(index)
        self.encoders.append(OneHotEncoder())
    
        temp = self.encoders[-1].transform(self.X[:,self.feature_indices[-1]])
        self.X = np.delete(self.X, self.feature_indices[-1], axis=1)
  
        for k in range(len(temp[0,:])):
            self.X = np.insert(self.X, self.feature_indices[-1]+k, temp[:,k], axis=1)
        
        self.fit = True
        return self.X  
    
    def transform(self, X):
        import numpy as np
        self.X = np.copy(X)
        
        if self.fit == True:
            for i in range(len(self.feature_indices)):
                temp = self.encoders[i].transform(self.X[:,self.feature_indices[i]])
                self.X = np.delete(self.X, self.feature_indices[i], axis=1)
                for k in range(len(temp[0,:])):
                    self.X = np.insert(self.X, self.feature_indices[i]+k, temp[:,k], axis=1)
            return self.X     
        
        else:
            raise RuntimeError("encoder not fit")

class SimpleImputer():
    def transform(self, X, column, missing_values="nan", strategy="mean"):
        import numpy as np
        self.X = np.copy(X)
        self.column = column
        self.missing_values = missing_values
        self.strategy = strategy
        
        if self.strategy != "mean":
            raise ValueError("invalid value for 'strategy'")
            
        if self.strategy == "mean":
            if self.X[:,self.column].dtype == "<U4":
                raise TypeError("invalid type for 'X[:,i]'")
            
            if self.missing_values == "nan":
                mean = np.nanmean(np.array(self.X[:,self.column], dtype=np.float64))
            else:
                def ColMean():
                    values = list(self.X[:,self.column].flatten())
                    for i in range(len(values)):
                        if i+1 > len(values):
                            break
                        if values[i] == self.missing_values:
                            values.pop(i)
                    mean = np.mean(values)
                    return mean
                
                mean = ColMean()
            
            for i in range(len(self.X[:,0])):
                if self.missing_values == "nan":
                    if np.isnan(self.X[i,self.column]):
                        self.X[i,self.column] = float(mean)
                else:
                    if self.X[i,self.column] == self.missing_values:
                        self.X[i,self.column] = float(mean)
            return self.X

class DGImputer():
    def evaluate(self, pos):
        import numpy as np
        self.diffs = np.zeros(shape=self.X.shape)
        self.scores = np.zeros(shape=(self.X.shape[0], 1))
        for i in range(len(self.X_copy[0,:])):
            self.diffs[pos[0], i] = self.X_copy[pos[0], i]
            
        for i in range(len(self.X_copy[:,0])):
            for j in range(len(self.X_copy[0,:])):
                if j != pos[1]: 
                    self.diffs[i,j] = np.absolute(self.X_copy[pos[0], j] - self.X_copy[i,j])
                    
        for i in range(self.scores.shape[0]):
            self.scores[i] = np.sum(self.diffs[i])
        
        return [np.argsort(self.scores, axis=0)[i] for i in range(self.scores.shape[0]//2) if np.isnan(self.X_copy[np.argsort(self.scores, axis=0)[i][0], self.col]) == False]
        
    def transform(self, X, column):
        import numpy as np
        self.X_copy = np.copy(X)
        self.X = np.copy(X)
        self.col = column
        self.nans = []
        self.diffs = np.zeros(shape=self.X.shape)
        self.scores = np.zeros(shape=(self.X.shape[0], 1))
        self.dg = []
        
        for i in range(len(self.X[0,:])):
            if type(self.X[0,i]) == str:
                for j in range(len(self.X[:,i])):
                    self.X[j,i] = 0
        
        for i in range(len(self.X[:,self.col])):
            if np.isnan(self.X[i,self.col]):
                self.nans.append((i,self.col))
        
        for i in range(len(self.nans)):
            if np.nanstd(self.X[:,self.col]) > 5:
                self.dg = self.evaluate(self.nans[i])
                self.dg_vals = [self.X[self.evaluate(self.nans[i]), self.col]]
                self.dg_mean = np.mean(np.array([self.dg_vals[0][0], self.dg_vals[0][1], self.dg_vals[0][2]]))
                self.X[self.nans[i][0], self.nans[i][1]] = self.dg_mean
            else:
                self.dg = self.evaluate(self.nans[i])
                self.X[self.nans[i][0], self.nans[i][1]] = self.X[self.dg[0][0], self.col]
           
        return self.X