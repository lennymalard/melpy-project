import numpy as np

class RobustScaler:
    """
    A class for scaling features using statistics that are robust to outliers.

    Attributes
    ----------
    fitted : bool
        Indicates whether the scaler has been fitted to the data.
    q1_list : list
        List of the first quartile (25th percentile) values for each feature.
    median_list : list
        List of the median (50th percentile) values for each feature.
    q3_list : list
        List of the third quartile (75th percentile) values for each feature.

    Methods
    -------
    fit(X : ndarray)
        Computes the quartiles and median for scaling.
    transform(X : ndarray)
        Scales the data using the computed quartiles and median.
    """
    def __init__(self):
        """
        Initializes the RobustScaler object.
        """
        self.fitted = False
        self.q1_list, self.median_list, self.q3_list = [], [], []

    def fit(self, X):
        """
        Computes the quartiles and median for scaling.

        Parameters
        ----------
        X : ndarray
            The input data to fit the scaler.

        Raises
        ------
        ValueError
            If the input data has more than 2 dimensions.
        """
        if X.ndim > 2:
            raise ValueError("scalers don't support more than 2 dimensions.")
        self.q1_list = np.quantile(X, 0.25, axis=0)
        self.median_list = np.quantile(X, 0.5, axis=0)
        self.q3_list = np.quantile(X, 0.75, axis=0)
        self.fitted = True

    def transform(self, X):
        """
        Scales the data using the computed quartiles and median.

        Parameters
        ----------
        X : ndarray
            The input data to be scaled.

        Returns
        -------
        X_scaled : ndarray
            The scaled data.
        """
        if not self.fitted:
            self.fit(X)
        X_scaled = (X - self.median_list) / (self.q3_list - self.q1_list + 1e-5)
        return X_scaled

class MinMaxScaler:
    """
    A class for scaling features to a given range.

    Attributes
    ----------
    fitted : bool
        Indicates whether the scaler has been fitted to the data.
    min_list : list
        List of the minimum values for each feature.
    max_list : list
        List of the maximum values for each feature.

    Methods
    -------
    fit(X : ndarray)
        Computes the minimum and maximum values for scaling.
    transform(X : ndarray)
        Scales the data using the computed minimum and maximum values.
    """
    def __init__(self):
        """
        Initializes the MinMaxScaler object.
        """
        self.fitted = False
        self.min_list, self.max_list = [], []

    def fit(self, X):
        """
        Computes the minimum and maximum values for scaling.

        Parameters
        ----------
        X : ndarray
            The input data to fit the scaler.

        Raises
        ------
        ValueError
            If the input data has more than 2 dimensions.
        """
        if X.ndim > 2:
            raise ValueError("scalers don't support more than 2 dimensions")
        self.min_list = np.min(X, axis=0)
        self.max_list = np.max(X, axis=0)
        self.fitted = True

    def transform(self, X):
        """
        Scales the data using the computed minimum and maximum values.

        Parameters
        ----------
        X : ndarray
            The input data to be scaled.

        Returns
        -------
        X_scaled : ndarray
            The scaled data.
        """
        if not self.fitted:
            self.fit(X)
        X_scaled = (X - self.min_list) / (self.max_list - self.min_list + 1e-5)
        return X_scaled

class StandardScaler:
    """
    A class for standardizing features by removing the mean and scaling to unit variance.

    Attributes
    ----------
    fitted : bool
        Indicates whether the scaler has been fitted to the data.
    mean_list : list
        List of the mean values for each feature.
    std_list : list
        List of the standard deviation values for each feature.

    Methods
    -------
    fit(X : ndarray)
        Computes the mean and standard deviation for scaling.
    transform(X : ndarray)
        Scales the data using the computed mean and standard deviation.
    """
    def __init__(self):
        """
        Initializes the StandardScaler object.
        """
        self.fitted = False
        self.mean_list, self.std_list = [], []

    def fit(self, X):
        """
        Computes the mean and standard deviation for scaling.

        Parameters
        ----------
        X : ndarray
            The input data to fit the scaler.

        Raises
        ------
        ValueError
            If the input data has more than 2 dimensions.
        """
        if X.ndim > 2:
            raise ValueError("scalers don't support more than 2 dimensions")
        self.mean_list = np.mean(X, axis=0)
        self.std_list = np.std(np.array(X, dtype=np.float64), axis=0)
        self.fitted = True

    def transform(self, X):
        """
        Scales the data using the computed mean and standard deviation.

        Parameters
        ----------
        X : ndarray
            The input data to be scaled.

        Returns
        -------
        X_scaled : ndarray
            The scaled data.
        """
        if not self.fitted:
            self.fit(X)
        X_scaled = (X - self.mean_list) / (self.std_list + 1e-5)
        return X_scaled

class OneHotEncoder:
    """
    A class for encoding categorical features as a one-hot numeric array.

    Attributes
    ----------
    fitted : bool
        Indicates whether the encoder has been fitted to the data.
    categories : list
        List of unique categories found in the data.

    Methods
    -------
    fit(y : ndarray)
        Fits the encoder to the data by identifying unique categories.
    transform(y : ndarray)
        Transforms the data into a one-hot encoded array.
    """
    def __init__(self):
        """
        Initializes the OneHotEncoder object.
        """
        self.fitted = False
        self.categories = []

    def fit(self, y):
        """
        Fits the encoder to the data by identifying unique categories.

        Parameters
        ----------
        y : ndarray
            The input data to fit the encoder.
        """
        y = np.asarray(y).flatten()
        self.categories = np.unique(y)
        self.fitted = True

    def transform(self, y):
        """
        Transforms the data into a one-hot encoded array.

        Parameters
        ----------
        y : ndarray
            The input data to be transformed.

        Returns
        -------
        result : ndarray
            The one-hot encoded array.
        """
        if not self.fitted:
            self.fit(y)
        y = np.asarray(y).flatten()
        result = np.zeros((len(y), len(self.categories)))
        for i, category in enumerate(self.categories):
            result[:, i] = (y == category)
        return np.array(result, dtype=np.float64)

class StringEncoder:
    """
    A class for encoding string features using one-hot encoding.

    Attributes
    ----------
    feature_indices : list
        List of indices of string features.
    encoders : list
        List of OneHotEncoder objects for each string feature.
    fitted : bool
        Indicates whether the encoder has been fitted to the data.

    Methods
    -------
    transform(X : ndarray)
        Transforms the data by encoding string features.
    fit(X : ndarray)
        Fits the encoder to the data by identifying string features and fitting OneHotEncoder objects.
    _encode(X : ndarray)
        Encodes the string features using the fitted OneHotEncoder objects.
    """
    def __init__(self):
        """
        Initializes the StringEncoder object.
        """
        self.feature_indices = []
        self.encoders = []
        self.fitted = False

    def transform(self, X):
        """
        Transforms the data by encoding string features.

        Parameters
        ----------
        X : ndarray
            The input data to be transformed.

        Returns
        -------
        X_encoded : ndarray
            The transformed data with string features encoded.
        """
        if not self.fitted:
            self.fit(X)
        return self._encode(X)

    def fit(self, X):
        """
        Fits the encoder to the data by identifying string features and fitting OneHotEncoder objects.

        Parameters
        ----------
        X : ndarray
            The input data to fit the encoder.
        """
        num_features = X.shape[1]
        for i in range(num_features):
            if isinstance(X[0, i], str):
                encoder = OneHotEncoder()
                encoder.fit(X[:, i])
                self.feature_indices.append(i)
                self.encoders.append(encoder)
        self.fitted = True

    def _encode(self, X):
        """
        Encodes the string features using the fitted OneHotEncoder objects.

        Parameters
        ----------
        X : ndarray
            The input data to be encoded.

        Returns
        -------
        X_encoded : ndarray
            The encoded data.
        """
        for i, encoder in zip(self.feature_indices, self.encoders):
            transformed = encoder.transform(X[:, i])
            X = np.delete(X, i, axis=1)
            X = np.hstack((X[:, :i], transformed, X[:, i:]))
        return np.array(X, dtype=np.float64)

class SimpleImputer:
    """
    A class for imputing missing values in a dataset.

    Methods
    -------
    transform(X, column, missing_values=np.nan, strategy="mean")
        Imputes missing values in the specified column using the specified strategy.
    """
    def transform(self, X, column, missing_values=np.nan, strategy="mean"):
        """
        Imputes missing values in the specified column using the specified strategy.

        Parameters
        ----------
        X : ndarray
            The input data.
        column : int
            The index of the column to impute.
        missing_values : float or str, optional
            The value to consider as missing. Default is np.nan.
        strategy : str, optional
            The imputation strategy. Currently, only "mean" is supported. Default is "mean".

        Returns
        -------
        X_copy : ndarray
            The input data with missing values imputed.

        Raises
        ------
        ValueError
            If the strategy is not one of ("mean").
        TypeError
            If the column data type is not numeric.
        """
        X_copy = np.copy(X)

        if strategy != "mean":
            raise ValueError("`strategy` must be one of ('mean').")

        if X_copy[:, column].dtype == "<U4":
            raise TypeError("`column` must be numeric.")

        if np.isnan(missing_values):
            mask = np.isnan(X_copy[:, column].astype(float))
            mean = np.nanmean(X_copy[:, column].astype(float))
        else:
            mask = (X_copy[:, column] == missing_values)
            filtered_values = X_copy[~mask, column].astype(float)
            mean = np.mean(filtered_values) if filtered_values.size > 0 else 0

        X_copy[mask, column] = mean
        return X_copy
