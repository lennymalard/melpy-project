# TODO
# Rajouter KNN Imputer
# Vérifier à nouveau chaque classe
# Features Encoder ???

import numpy as np

class RobustScaler:
    def __init__(self):
        self.fitted = False
        self.q1_list, self.median_list, self.q3_list = [], [], []
    
    def fit(self, X):
        if X.ndim > 2:
            raise ValueError("scalers don't support more than 2 dimensions")
        self.q1_list = np.quantile(X, 0.25, axis=0)
        self.median_list = np.quantile(X, 0.5, axis=0)
        self.q3_list = np.quantile(X, 0.75, axis=0)
        self.fitted = True
        
    def transform(self, X):
        if not self.fitted:
            self.fit(X)
        X_scaled = (X - self.median_list) / (self.q3_list - self.q1_list + 1e-5)
        return X_scaled

class MinMaxScaler:
    def __init__(self):
        self.fitted = False
        self.min_list, self.max_list = [], []
    
    def fit(self, X):
        if X.ndim > 2:
            raise ValueError("scalers don't support more than 2 dimensions")
        self.min_list = np.min(X, axis=0)
        self.max_list = np.max(X, axis=0)
        self.fitted = True
        
    def transform(self, X):
        if not self.fitted:
            self.fit(X)
        X_scaled = (X - self.min_list) / (self.max_list - self.min_list + 1e-5)
        return X_scaled

class StandardScaler:
    def __init__(self):
        self.fitted = False
        self.mean_list, self.std_list = [], []
    
    def fit(self, X):
        if X.ndim > 2:
            raise ValueError("scalers don't support more than 2 dimensions")
        self.mean_list = np.mean(X, axis=0)
        self.std_list = np.std(np.array(X, dtype =np.float64), axis=0)
        self.fitted = True
        
    def transform(self, X):
        if not self.fitted:
            self.fit(X)
        X_scaled = (X - self.mean_list) / (self.std_list + 1e-5)
        return X_scaled

class OneHotEncoder:
    def __init__(self):
        self.fitted = False
        self.categories = []

    def fit(self, y):
        y = np.asarray(y).flatten()
        self.categories = np.unique(y)
        self.fitted = True

    def transform(self, y):
        if not self.fitted:
            self.fit(y)
        y = np.asarray(y).flatten()
        result = np.zeros((len(y), len(self.categories)))
        for i, category in enumerate(self.categories):
            result[:, i] = (y == category)
        return np.array(result, dtype=np.float64)

class StringEncoder:
    def __init__(self):
        self.feature_indices = []
        self.encoders = []
        self.fitted = False

    def transform(self, X):
        if not self.fitted:
            self.fit(X)
        return self._encode(X)
    
    def fit(self, X):
        num_features = X.shape[1]
        for i in range(num_features):
            if isinstance(X[0, i], str):
                encoder = OneHotEncoder()
                encoder.fit(X[:, i])
                self.feature_indices.append(i)
                self.encoders.append(encoder)
        self.fitted = True

    def _encode(self, X):
        for i, encoder in zip(self.feature_indices, self.encoders):
            transformed = encoder.transform(X[:, i])
            X = np.delete(X, i, axis=1)
            X = np.hstack((X[:, :i], transformed, X[:, i:]))
        return np.array(X, dtype=np.float64)

class SimpleImputer:
    def transform(self, X, column, missing_values=np.nan, strategy="mean"):
        X_copy = np.copy(X)
        
        if strategy != "mean":
            raise ValueError("invalid value for 'strategy'")
        
        if X_copy[:, column].dtype == "<U4":
            raise TypeError("invalid type for 'column'")
        
        if np.isnan(missing_values):
            mask = np.isnan(X_copy[:, column].astype(float))
            mean = np.nanmean(X_copy[:, column].astype(float))
        else:
            mask = (X_copy[:, column] == missing_values)
            filtered_values = X_copy[~mask, column].astype(float)
            mean = np.mean(filtered_values) if filtered_values.size > 0 else 0
        
        X_copy[mask, column] = mean
        return X_copy