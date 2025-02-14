import numpy as np
import re
import itertools
import json

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
        for i, encoder in zip(reversed(self.feature_indices), reversed(self.encoders)):
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


class Tokenizer:
    """
    A text tokenizer supporting word-level and character-level tokenization.

    Parameters
    ----------
    strategy : str, optional
        Tokenization strategy - 'word' or 'character'. Default is 'word'.
    lower : bool, optional
        Whether to convert text to lowercase. Default is True.

    Attributes
    ----------
    strategy : str
        Active tokenization strategy ('word' or 'character').
    lower : bool
        Lowercasing status.
    value_index : dict
        Mapping from token values to indices.
    index_value : dict
        Mapping from indices to token values.
    fitted : bool
        Whether the tokenizer has been fitted on data.

    Raises
    ------
    TypeError
        If strategy is not a string.
    ValueError
        If strategy is not 'word' or 'character'.
    """
    def __init__(self, strategy, lower=True):
        """
       Initialize tokenizer with specified strategy and text processing options.
       """

        if not isinstance(strategy, str):
            raise TypeError("'strategy' must be of type str.")

        if strategy.lower() not in ['word', 'character', 'char']:
            raise ValueError(f"'strategy' must be {', '.join([str(e) for e in ['word', 'character', 'char']])}.")

        self.strategy = strategy.lower()
        self.lower = lower
        self.value_index = {}
        self.index_value = {}
        self.encoder = OneHotEncoder()
        self.fitted = False

    def merge_lists(self, lists):
        """
        Flatten a list of lists into a single list.

        Parameters
        ----------
        lists : list of lists
            Nested list structure to flatten

        Returns
        -------
        list
            Flattened list containing all elements

        Raises
        ------
        TypeError
            If input is not a list of lists
        """
        if not isinstance(lists, list) and not isinstance(lists[0], list):
            raise TypeError("'lists' must be a list of lists.")

        return list(itertools.chain.from_iterable(lists))

    def word_tokenize(self, text):
        """
        Tokenize text into words with advanced pattern matching.

        Parameters
        ----------
        text : str
            Input text to tokenize

        Returns
        -------
        list
            List of word tokens

        Raises
        ------
        TypeError
            If input is not a string

        Notes
        -----
        Handles:
        - Abbreviations (including accented characters)
        - Currency/numbers/percentages
        - Hyphenated numbers and words
        - Words with apostrophes and accents
        - Special characters and whitespace
        """

        if not isinstance(text, str):
            raise TypeError("'text' must be of type str.")

        text = text.lower() if self.lower else text
        pattern = r"""
                (?:[A-Z\xC0-\xD6\xD8-\xDE]\.)+    # Abbreviations (including accented uppercase)
                | <[A-Za-z0-9_]+>                 # Special tokens like <PAD>, <EOS>
                | \$?\d+(?:[.,]\d+)*%?            # Currency, numbers, percentages
                | \d+(?:-\d+)+                    # Hyphenated numbers (e.g., dates)
                | [a-zA-Z\xC0-\xFF]+(?:[-'’][a-zA-Z\xC0-\xFF]*)* # Words with accents, hyphens, apostrophes
                | [_]                             # Underscores
                | [\n\t\r\f\v]                    # Whitespace characters
                | [^\w\s]                         # Punctuation
            """
        tokens = re.findall(pattern, text, re.VERBOSE | re.IGNORECASE)
        return tokens

    def char_tokenize(self, text):
        """
        Tokenize text into individual characters.

        Parameters
        ----------
        text : str
            Input text to tokenize

        Returns
        -------
        list
            List of character tokens

        Raises
        ------
        TypeError
            If input is not a string
        """
        if not isinstance(text, str):
            raise TypeError("'text' must be of type str.")

        text = text.lower() if self.lower else text
        return list(text)

    def mapping(self, tokens):
        """
        Create bidirectional token-index mappings.

        Parameters
        ----------
        tokens : list
           List of tokens to create mappings from

        Returns
        -------
        tuple
           (index_value mapping, value_index mapping)

        Raises
        ------
        TypeError
           If tokens is not a list
        """
        if not isinstance(tokens, list):
            raise TypeError("'tokens' must be of type list.")

        index_value = {index: value for index, value in enumerate(sorted(set(tokens)))}
        value_index = {value: index for index, value in enumerate(sorted(set(tokens)))}

        return index_value, value_index

    def fit_on_texts(self, texts):
        """
        Learn vocabulary from list of texts.

        Parameters
        ----------
        texts : list of str
            Training texts to learn vocabulary from

        Raises
        ------
        TypeError
            If input is not a list of strings

        Notes
        -----
        - Creates vocabulary based on selected strategy
        - Stores mappings in index_value and value_index attributes
        - Sets fitted flag to True
        """
        if not isinstance(texts, list):
            raise TypeError("'texts' must be of type list.")

        self.fitted = True
        if self.strategy == "word":
            word_tokens = self.merge_lists([self.word_tokenize(text) for text in texts])
            self.index_value, self.value_index = self.mapping(word_tokens)

        elif self.strategy == "character":
            char_tokens = self.merge_lists([self.char_tokenize(text) for text in texts])
            self.index_value, self.value_index = self.mapping(char_tokens)

    def texts_to_sequences(self, texts):
        """
        Convert input texts to sequences of indices.

        Parameters
        ----------
        texts : str or list of str
            Text(s) to convert to indices

        Returns
        -------
        list of lists
            Sequence(s) of token indices

        Raises
        ------
        TypeError
            If input is not string or list of strings

        Notes
        -----
        - Automatically fits tokenizer if not already fitted
        - Handles both single strings and lists of strings
        """
        if not isinstance(texts, list) and isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list) and not isinstance(texts, str):
            raise TypeError("'texts' must be either a string or a list of strings.")

        if not self.fitted:
            self.fit_on_texts(texts)

        if self.strategy == "word":
            texts_tokenized = [self.word_tokenize(text) for text in texts]

        elif self.strategy in ("character", "char"):
            texts_tokenized = [self.char_tokenize(text) for text in texts]

        return [[self.value_index[value] for value in text] for text in texts_tokenized]

    def save_vocabulary(self, filename):
        """
        Serialize the vocabulary mappings to a JSON file.

        Saves both index-to-value (index_value) and value-to-index (value_index)
        mappings to a human-readable JSON file. Useful for preserving vocabulary state
        between sessions.

        Arguments:
        ----------
            filename (str): Path/filename where vocabulary will be saved (.json recommended)

        Example:
        --------
            > vocab.save_vocabulary("vocab.json")

        Note:
        -----
            - Creates the file if it doesn't exist, overwrites if it does
            - Uses JSON format with 4-space indentation for readability
        """
        data = {"index_value": self.index_value, "value_index": self.value_index}
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Data saved to {filename}")

    def load_vocabulary(self, path):
        """
        Deserialize vocabulary mappings from a JSON file.

        Restores vocabulary state from a previously saved file, including:
        - index-to-value mapping
        - value-to-index mapping
        - Sets 'fitted' flag to True

        Arguments:
        ----------
            path (str): Path to valid vocabulary JSON file

        Example:
        --------
            > vocab.load_vocabulary("vocab.json")

        Raises:
        -------
            FileNotFoundError: If specified file doesn't exist
            JSONDecodeError: If file contains invalid JSON
            KeyError: If required fields are missing in the JSON

        Note:
        -----
            - Requires file format matching save_vocabulary() output
            - Overwrites any existing vocabulary mappings in memory
        """
        with open(path, 'r') as file:
            data = json.load(file)
            self.index_value = {int(index): value for index, value in data["index_value"].items()}
            self.value_index = {value: int(index) for value, index in data["value_index"].items()}
            self.fitted = True
        print(f"Data loaded from {path}")

    def one_hot_encode(self, token):
        """
        One-hot encode the given token(s).

        This method converts the input token(s) into their corresponding one-hot encoded
        representation using a fitted encoder. If the encoder is not already fitted, it
        is fitted using the keys from the `index_value` dictionary.

        Parameters
        ----------
        token : int, str, list, or numpy.ndarray
            The token(s) to be one-hot encoded. If the token is a string, it is first
            converted into a numpy array before encoding. Lists and numpy arrays are
            processed element-wise.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array where each row corresponds to the one-hot encoded
            representation of the input token(s).

        Raises
        ------
        TypeError
            If the input `token` is not of type `int`, `str`, `list`, or `numpy.ndarray`.

        Notes
        -----
        - The encoder is fitted on the keys of the `index_value` dictionary if it has not
          been fitted previously.
        - For string tokens, the method assumes the encoder is compatible with the values
          derived from `index_value` keys. Ensure the encoder's categories match the
          expected input tokens.

        Examples
        --------
        Assuming `self.index_value` is properly initialized and the encoder is compatible:

        > Tokenizer.encoder.one_hot_encode(5)
        array([[0., 0., 0., 0., 1.]])

        > Tokenizer.encoder.one_hot_encode("apple")
        array([[1., 0., 0., 0., 0.]])

        > Tokenizer.encoder.one_hot_encode([1, 2, 3])
        array([[0., 1., 0.],
               [0., 0., 1.],
               [1., 0., 0.]])
        """
        if not self.encoder.fitted:
            self.encoder.fit(np.array(list(self.index_value.keys())))

        if isinstance(token, (int, str, list, np.ndarray)):
            return self.encoder.transform(np.array(token))
        elif isinstance(token, str):
            return self.encoder.transform(self.value_index[token])
        else:
            raise TypeError("'token' must be of type array, int or str.")

def generate_sequence_dataset(tokens, context_window=2):
    """
    Generate input sequences and target tokens for a sequence prediction task.

    Creates sliding windows of tokens as input sequences (X) and the subsequent token
    after each window as the target (Y). Used to prepare training data for models
    predicting the next token in a sequence (e.g., language modeling).

    Parameters
    ----------
    tokens : list or array-like
        A 1D sequence of tokens (integers, strings, or other hashable types).
    context_window : int, default=2
        Number of tokens to include on both sides of the center token in each input sequence.
        Total sequence length = 2 * context_window + 1.

    Returns
    -------
    x : numpy.ndarray
        A 2D array of shape (n_samples, sequence_length) where each row is a context window.
    y : numpy.ndarray
        A 1D array of shape (n_samples,) where each element is the target token following its sequence.

    Notes
    -----
    - The first valid sequence starts at index `context_window`.
    - The last valid sequence ends at index `len(tokens) - context_window - 2`.
    - If `len(tokens) < 2 * context_window + 2`, returns empty arrays.

    Example
    -------
    > tokens = [0, 1, 2, 3, 4, 5]
    > context_window = 1
    > x, y = generate_sequence_dataset(tokens, context_window)
    > x  # Input sequences (length = 2*1 + 1 = 3)
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])
    > y  # Targets (next token after each sequence)
    array([3, 4, 5])
    """
    x = []
    y = []
    for i in range(context_window, len(tokens)-context_window-1):
        sequence = tokens[i-context_window:i+context_window+1]
        x.append(sequence)
        y.append(tokens[i+context_window+1])
    return np.array(x), np.array(y)