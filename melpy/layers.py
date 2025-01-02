import numpy as np
from math import sqrt
from .im2col import *
from .tensor import *

class Layer:
    """
    Base class for all layers in the neural network.

    Attributes
    ----------
    inputs : Tensor
        Input data.
    outputs : Tensor
        Output data.
    dX : ndarray
        Partial derivative of loss with respect to input.
    dY : ndarray
        Partial derivative of loss with respect to output.

    Methods
    -------
    derivative()
        Computes the derivative of the layer.
    forward()
        Computes the forward pass of the layer.
    backward(dX : ndarray)
        Computes the backward pass of the layer.
    """
    def __init__(self):
        """
        Initializes the Layer class.
        """
        self.inputs = None
        self.outputs = None
        self.dX = None
        self.dY = None

    def derivative(self):
        """
        Computes the derivative of the layer.

        Returns
        -------
        ndarray
            The derivative of the layer.
        """
        pass

    def forward(self):
        """
        Computes the forward pass of the layer.

        Returns
        -------
        Tensor
            The output data.
        """
        pass

    def backward(self, dX):
        """
        Computes the backward pass of the layer.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of loss with respect to output data.

        Returns
        -------
        ndarray
            Partial derivative of loss with respect to input data.
        """
        pass

    def zero_grad(self):
        pass

class Activation:
    """
    Base class for all activation functions.
    """
    def __init__(self):
        """
        Initializes the Activation class.
        """
        pass

class Dense(Layer):
    """
    A class that performs dense (fully connected) layer operations.

    Attributes
    ----------
    inputs : Tensor
        Input data.
    outputs : Tensor
        Output data.
    dX : ndarray
        Partial derivative of loss with respect to input.
    dY : ndarray
        Partial derivative of loss with respect to output.
    dW : ndarray
        Partial derivative of loss with respect to weight.
    dB : ndarray
        Partial derivative of loss with respect to bias.
    weights : Tensor
        Weights of the dense layer.
    biases : Tensor
        Biases of the dense layer.
    weight_momentums : Tensor
        Momentum for weights (also known as first order momentum).
    bias_momentums : Tensor
        Momentum for biases (also known as first order momentum).
    weight_cache : Tensor
        Cache for weights (also known as second order momentum).
    bias_cache : Tensor
        Cache for biases (also known as second order momentum).

    Methods
    -------
    forward()
        Computes the forward pass of the dense layer.
    backward(dX : ndarray)
        Computes the backward pass of the dense layer.
    """
    def __init__(self, n_in, n_out, weight_initializer="he_uniform", activation=None):
        """
        Initializes the Dense layer.

        Parameters
        ----------
        n_in : int
            Number of input features.
        n_out : int
            Number of output features.
        weight_initializer : str, optional
            Weight initialization method ('he_normal', 'glorot_normal', 'he_uniform', 'glorot_uniform').
            Default is 'he_uniform'.
        activation : Activation, optional
            Activation function ('relu', 'leaky relu', 'sigmoid', 'softmax').

        Raises
        ------
        ValueError
            If the weight initialization method is invalid.
        """
        super().__init__()

        def initialize_weights(weight_init, n_in, n_out):
            """
            Initializes the weights based on the specified method.

            Parameters
            ----------
            weight_init : str
                Weight initialization method.
            n_in : int
                Number of input features.
            n_out : int
                Number of output features.

            Returns
            -------
            Tensor
                Initialized weights.
            """
            if weight_init == "he_normal":
                weights = Tensor(np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in))
            elif weight_init == "he_uniform":
                limit = np.sqrt(6 / n_in)
                weights = Tensor(np.random.uniform(-limit, limit, (n_in, n_out)))
            elif weight_init == "glorot_normal":
                weights = Tensor(np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out)))
            elif weight_init == "glorot_uniform":
                limit = np.sqrt(6 / (n_in + n_out))
                weights = Tensor(np.random.uniform(-limit, limit, (n_in, n_out)))
            else:
                raise ValueError(
                    "`weight_init` must be either 'he_uniform', 'he_normal', 'glorot_uniform' or 'glorot_normal'.")
            return weights

        self.weights = initialize_weights(weight_initializer, n_in, n_out)
        self.biases = Tensor(np.random.rand(1, n_out))
        self.dW = np.zeros_like(self.weights)
        self.dB = np.zeros_like(self.biases)
        self.weight_momentums = Tensor(np.zeros_like(self.weights))
        self.bias_momentums = Tensor(np.zeros_like(self.biases))
        self.weight_cache = Tensor(np.zeros_like(self.weights))
        self.bias_cache = Tensor(np.zeros_like(self.biases))
        self.activation = activation

        if not isinstance(activation, Activation) and activation is not None:
            raise TypeError("`activation` must be of type `Activation`.")

    def forward(self):
        """
        Computes the forward pass of the dense layer.

        Returns
        -------
        Tensor
            The output data.
        """
        self.outputs = dot(self.inputs, self.weights) + self.biases

        if self.activation is not None:
            self.activation.inputs = self.outputs
            return self.activation.forward()

        return self.outputs

    def backward(self, dX):
        """
        Computes the backward pass of the dense layer.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of loss with respect to output data.

        Returns
        -------
        ndarray
            Partial derivative of loss with respect to input data.
        """
        if self.activation is not None:
            dX = self.activation.backward(dX)

        self.dY = dX
        self.outputs.backward(self.dY)
        self.dX = self.inputs.grad
        self.dW = self.weights.grad
        return self.dX

    def zero_grad(self):
        if self.activation is not None:
            self.activation.zero_grad()
        self.outputs.zero_grad()

class ReLU(Layer, Activation):
    """
    A class that performs ReLU (Rectified Linear Unit) layer operations.

    Attributes
    ----------
    inputs : Tensor
        Input data.
    outputs : Tensor
        Output data.
    dX : ndarray
        Partial derivative of loss with respect to input.
    dY : ndarray
        Partial derivative of loss with respect to output.

    Methods
    -------
    derivative()
        Computes the ReLU derivative.
    forward()
        Computes the ReLU forward pass.
    backward(dX : ndarray)
        Computes the ReLU backward pass.
    """
    def __init__(self):
        """
        Initializes the ReLU layer.
        """
        super().__init__()

    def derivative(self):
        """
        Computes the ReLU derivative.

        Returns
        -------
        ndarray
            The differentiated input data.
        """
        dA = np.ones_like(self.inputs.array)
        dA[self.inputs.array <= 0] = 0
        return dA

    def forward(self):
        """
        Computes the ReLU forward pass.

        Returns
        -------
        Tensor
            The output data.
        """
        self.outputs = Tensor(np.maximum(0, self.inputs.array))
        return self.outputs

    def backward(self, dX):
        """
        Computes the ReLU backward pass.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of loss with respect to output data.

        Returns
        -------
        ndarray
            Partial derivative of loss with respect to input data.
        """
        self.dY = dX
        self.dX = self.dY * self.derivative()
        return self.dX

class LeakyReLU(Layer, Activation):
    """
    A class that performs Leaky ReLU layer operations.

    Attributes
    ----------
    inputs : Tensor
        Input data.
    outputs : Tensor
        Output data.
    dX : ndarray
        Partial derivative of loss with respect to input.
    dY : ndarray
        Partial derivative of loss with respect to output.

    Methods
    -------
    derivative()
        Computes the Leaky ReLU derivative.
    forward()
        Computes the Leaky ReLU forward pass.
    backward(dX : ndarray)
        Computes the Leaky ReLU backward pass.
    """
    def __init__(self):
        """
        Initializes the LeakyReLU layer.
        """
        super().__init__()

    def derivative(self):
        """
        Computes the Leaky ReLU derivative.

        Returns
        -------
        ndarray
            The differentiated input data with respect to loss.
        """
        dA = np.ones_like(self.inputs.array)
        dA[self.inputs.array <= 0] = 0.01
        return dA

    def forward(self):
        """
        Computes the Leaky ReLU forward pass.

        Returns
        -------
        Tensor
            The output data.
        """
        self.outputs = Tensor(np.where(self.inputs.array > 0, self.inputs.array, self.inputs.array * 0.01))
        return self.outputs

    def backward(self, dX):
        """
        Computes the Leaky ReLU backward pass.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of loss with respect to output data.

        Returns
        -------
        ndarray
            Partial derivative of loss with respect to input data.
        """
        self.dY = dX
        self.dX = self.dY * self.derivative()
        return self.dX

class Sigmoid(Layer, Activation):
    """
    A class that performs Sigmoid layer operations.

    Attributes
    ----------
    inputs : Tensor
        Input data.
    outputs : Tensor
        Output data.
    dX : ndarray
        Partial derivative of loss with respect to input.
    dY : ndarray
        Partial derivative of loss with respect to output.

    Methods
    -------
    derivative()
        Computes the Sigmoid derivative.
    forward()
        Computes the Sigmoid forward pass.
    backward(dX : ndarray)
        Computes the Sigmoid backward pass.
    """
    def __init__(self):
        """
        Initializes the Sigmoid layer.
        """
        super().__init__()

    def derivative(self):
        """
        Computes the Sigmoid derivative.

        Returns
        -------
        ndarray
            The Sigmoid derivative.
        """
        return self.outputs * (1 - self.outputs)

    def forward(self):
        """
        Computes the Sigmoid forward pass.

        Returns
        -------
        Tensor
            The output data.
        """
        self.outputs = 1 / (1 + exp(-self.inputs))
        return self.outputs

    def backward(self, dX):
        """
        Computes the Sigmoid backward pass.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of loss with respect to output data.

        Returns
        -------
        ndarray
            Partial derivative of loss with respect to input data.
        """
        self.dY = dX
        self.outputs.backward(self.dY)
        self.dX = self.inputs.grad
        return self.dX

    def zero_grad(self):
        self.outputs.zero_grad()

class Softmax(Layer, Activation):
    """
    A class that performs Softmax layer operations.

    Attributes
    ----------
    inputs : Tensor
        Input data.
    outputs : Tensor
        Output data.
    dX : ndarray
        Partial derivative of loss with respect to input.
    dY : ndarray
        Partial derivative of loss with respect to output.

    Methods
    -------
    forward()
        Computes the Softmax forward pass.
    backward(dX : ndarray)
        Computes the Softmax backward pass.
    """
    def __init__(self):
        """
        Initializes the Softmax layer.
        """
        super().__init__()

    def forward(self):
        """
        Computes the Softmax forward pass.

        Returns
        -------
        Tensor
            The output data.
        """
        exp_values = exp(self.inputs - max(self.inputs, axis=1, keepdims=True))
        probabilities = exp_values / sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities
        return self.outputs

    def backward(self, dX):
        """
        Computes the Softmax backward pass.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of loss with respect to output data.

        Returns
        -------
        ndarray
            Partial derivative of loss with respect to input data.
        """
        self.dY = dX
        self.outputs.backward(self.dY)
        self.dX = self.inputs.grad
        return self.dX

    def zero_grad(self):
        self.outputs.zero_grad()

class Convolution2D(Layer):
    """
    A class that performs 2D convolution layer operations.

    Attributes
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    weights : Tensor
        Filters of the convolution layer.
    padding : str
        Padding type ('valid' or 'same').
    kernel_size : int
        Size of the convolution kernel.
    stride : int
        Stride of the convolution.
    weight_momentums : Tensor
        Momentum for weights.
    weight_cache : Tensor
        Cache for weights.
    biases : Tensor
        Biases of the convolution layer.
    bias_momentums : Tensor
        Momentum for biases.
    bias_cache : Tensor
        Cache for biases.

    Methods
    -------
    calculate_padding()
        Calculates the padding for the input.
    explicit_padding()
        Applies explicit padding to the input.
    get_output_size(input_height : int, input_width : int)
        Gets the output size of the convolution.
    forward()
        Computes the forward pass of the convolution layer.
    backward(dX : ndarray)
        Computes the backward pass of the convolution layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding="valid", stride=1, weight_initializer="he_uniform", use_bias=True, activation=None):
        """
        Initializes the Convolution2D layer.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Size of the convolution kernel.
        padding : str, optional
            Padding type ('valid' or 'same'). Default is 'valid'.
        stride : int, optional
            Stride of the convolution. Default is 1.
        weight_initializer : str, optional
            Weight initialization method ('he_normal', 'glorot_normal', 'he_uniform', 'glorot_uniform').
            Default is 'he_uniform'.
        activation : Activation, optional
            Activation function. Default is None.

        Raises
        ------
        TypeError
            If `padding` is not a string.
        ValueError
            If `padding` is not 'valid' or 'same'.
        """
        super().__init__()

        def initialize_weights(weight_init, in_channels, out_channels, kernel_size):
            """
            Initializes the weights based on the specified method.

            Parameters
            ----------
            weight_init : str
                Weight initialization method.
            in_channels : int
                Number of input channels.
            out_channels : int
                Number of output channels.
            kernel_size : int
                Size of the convolution kernel.

            Returns
            -------
            Tensor
                Initialized weights.
            """
            if weight_init == "he_normal":
                weights = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(
                    2.0 / (in_channels * kernel_size ** 2)))
            elif weight_init == "he_uniform":
                limit = np.sqrt(6 / (in_channels * kernel_size ** 2))
                weights = Tensor(
                    np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size)))
            elif weight_init == "glorot_normal":
                weights = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(
                    2.0 / (in_channels * kernel_size ** 2 + out_channels * kernel_size ** 2)))
            elif weight_init == "glorot_uniform":
                limit = np.sqrt(6 / (in_channels * kernel_size ** 2 + out_channels * kernel_size ** 2))
                weights = Tensor(
                    np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size)))
            else:
                raise ValueError(
                    "`weight_init` must be either 'he_uniform', 'he_normal', 'glorot_uniform' or 'glorot_normal'.")
            return weights

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = initialize_weights(weight_initializer, in_channels, out_channels, kernel_size)
        self.dW = np.zeros_like(self.weights)
        self.weight_momentums = Tensor(np.zeros_like(self.weights))
        self.weight_cache = Tensor(np.zeros_like(self.weights))
        if use_bias is True:
            self.biases = Tensor(np.zeros(shape=(1, out_channels, 1, 1)).astype(np.float64))
            self.dB = np.zeros_like(self.biases)
            self.bias_momentums = Tensor(np.zeros_like(self.biases))
            self.bias_cache = Tensor(np.zeros_like(self.biases))
        else:
            self.biases = None
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation

        if not isinstance(activation, Activation) and activation is not None:
            raise TypeError("`activation` must be of type `Activation`.")

        if not isinstance(padding, str):
            raise TypeError("`padding` must be a string.")
        if self.padding not in ("valid", "same"):
            raise ValueError("`padding` must be 'valid' or 'same'.")

        if self.padding == "same":
            self.stride = 1

    def calculate_padding(self):
        """
        Calculates the padding for the input.

        Returns
        -------
        tuple
            Padding values (top, bottom, left, right).
        """
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

            pad_top = pad_along_height.array.astype(int) // 2
            pad_bottom = pad_along_height.array.astype(int) - pad_top
            pad_left = pad_along_width.array.astype(int) // 2
            pad_right = pad_along_width.array.astype(int) - pad_left

            return (pad_top, pad_bottom, pad_left, pad_right)

    def explicit_padding(self):
        """
        Applies explicit padding to the input.

        Returns
        -------
        Tensor
            Padded input data.
        """
        pad_top, pad_bottom, pad_left, pad_right = self.calculate_padding()
        return Tensor(np.pad(self.inputs.array, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant'))

    def get_output_size(self, input_height, input_width):
        """
        Gets the output size of the convolution.

        Parameters
        ----------
        input_height : int
            Height of the input.
        input_width : int
            Width of the input.

        Returns
        -------
        tuple
            Output height and width.
        """
        if self.padding == 'valid':
            output_height = (input_height - self.kernel_size) // self.stride + 1
            output_width = (input_width - self.kernel_size) // self.stride + 1
        elif self.padding == 'same':
            output_height = np.ceil(input_height / self.stride)
            output_width = np.ceil(input_width / self.stride)

        return int(output_height), int(output_width)

    def forward(self):
        """
        Computes the forward pass of the convolution layer.

        Returns
        -------
        Tensor
            The output data.
        """
        self.input_padded = self.explicit_padding()

        self.input_cols = im2col(self.input_padded, self.kernel_size, self.stride)
        self.filter_cols = Tensor(self.weights.array.reshape(self.out_channels, -1))

        output_height, output_width = self.get_output_size(self.inputs.shape[2], self.inputs.shape[3])

        self.output_cols = self.filter_cols @ self.input_cols

        self.outputs = Tensor(np.array(np.hsplit(self.output_cols.array, self.inputs.shape[0])).reshape(
            (self.input_padded.shape[0], self.out_channels, output_height, output_width)
        ))

        if self.biases is not None:
            self.outputs = self.outputs + self.biases

        if self.activation is not None:
            self.activation.inputs = self.outputs
            return self.activation.forward()

        return self.outputs

    def backward(self, dX):
        """
        Computes the backward pass of the convolution layer.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of loss with respect to output data.

        Returns
        -------
        ndarray
            Partial derivative of loss with respect to input data.
        """
        if self.activation is not None:
            dX = self.activation.backward(dX)

        self.dY = dX

        self.dY_reshaped = self.dY.reshape(self.dY.shape[0] * self.dY.shape[1], self.dY.shape[2] * self.dY.shape[3])
        self.dY_reshaped = np.array(np.vsplit(self.dY_reshaped, self.inputs.shape[0]))
        self.dY_reshaped = np.concatenate(self.dY_reshaped, axis=-1)

        self.output_cols.backward(self.dY_reshaped)

        self.dX_cols = self.input_cols.grad
        self.dW_cols = self.filter_cols.grad

        self.dX_padded = col2im(Tensor(self.dX_cols), self.input_padded.shape, self.kernel_size, self.stride)

        if self.padding == "same":
            (pad_top, pad_bottom, pad_left, pad_right) = self.calculate_padding()
            self.dX = self.dX_padded.array[:, :, pad_top:-pad_bottom, pad_left:-pad_right]
        else:
            self.dX = self.dX_padded.array

        self.dW = self.dW_cols.reshape((self.dW_cols.shape[0], self.in_channels, self.kernel_size, self.kernel_size))

        self.outputs.backward(self.dY)
        self.dB = self.biases.grad

        return self.dX

    def zero_grad(self):
        if self.activation is not None:
            self.activation.zero_grad()
        self.output_cols.zero_grad()

class Pooling2D(Layer):
    """
    A class that performs 2D pooling layer operations.

    Attributes
    ----------
    pool_size : int
        Size of the pooling window.
    stride : int
        Stride of the pooling.
    mode : str
        Pooling mode ('max').

    Methods
    -------
    forward()
        Computes the forward pass of the pooling layer.
    backward(dX : ndarray)
        Computes the backward pass of the pooling layer.
    """
    def __init__(self, pool_size, stride, mode="max"):
        """
        Initializes the Pooling2D layer.

        Parameters
        ----------
        pool_size : int
            Size of the pooling window.
        stride : int
            Stride of the pooling.
        mode : str, optional
            Pooling mode ('max'). Default is 'max'.

        Raises
        ------
        ValueError
            If the pooling mode is invalid.
        """
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode

        if self.mode not in ("max"):
            raise ValueError("`mode` must be 'max'.")

    def forward(self):
        """
        Computes the forward pass of the pooling layer.

        Returns
        -------
        Tensor
            The output data.
        """
        output_height = int((self.inputs.shape[2] - self.pool_size + self.stride) // self.stride)
        output_width = int((self.inputs.shape[3] - self.pool_size + self.stride) // self.stride)

        output_shape = (self.inputs.shape[0], self.inputs.shape[1], output_height, output_width)

        self.input_cols = im2col(self.inputs, self.pool_size, self.stride)
        self.input_cols_reshaped = Tensor(
            np.array(np.hsplit(np.array(np.hsplit(self.input_cols.array, self.inputs.shape[0])), self.inputs.shape[1])))

        self.maxima = max(self.input_cols_reshaped, axis=2)
        self.maxima_reshaped = self.maxima.array.reshape(self.inputs.shape[1], -1)

        self.outputs = col2im(Tensor(self.maxima_reshaped), output_shape, 1, 1)

        return self.outputs

    def backward(self, dX):
        """
        Computes the backward pass of the pooling layer.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of loss with respect to output data.

        Returns
        -------
        ndarray
            Partial derivative of loss with respect to input data.
        """
        self.dY = dX
        self.dX = np.zeros_like(self.inputs)

        self.dY_cols = im2col(Tensor(self.dY), 1, 1)
        self.dY_cols_reshaped = np.array(
            np.hsplit(np.array(np.hsplit(self.dY_cols.array, self.dY.shape[0])), self.dY.shape[1]))

        self.maxima.backward(self.dY_cols_reshaped)

        self.dX_cols = np.concatenate(np.concatenate(self.input_cols_reshaped.grad, axis=1), axis=1)

        self.dX = col2im(Tensor(self.dX_cols), self.inputs.shape, self.pool_size, self.stride).array

        return self.dX

    def zero_grad(self):
        self.maxima.zero_grad()

class Flatten(Layer):
    """
    A class that performs flattening layer operations.

    Methods
    -------
    forward()
        Computes the forward pass of the flattening layer.
    backward(dX : ndarray)
        Computes the backward pass of the flattening layer.
    """
    def __init__(self):
        """
        Initializes the Flatten layer.
        """
        super().__init__()

    def forward(self):
        """
        Computes the forward pass of the flattening layer.

        Returns
        -------
        Tensor
            The output data.
        """
        self.outputs = Tensor(self.inputs.array.reshape((self.inputs.shape[0], -1)))
        return self.outputs

    def backward(self, dX):
        """
        Computes the backward pass of the flattening layer.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of loss with respect to output data.

        Returns
        -------
        ndarray
            Partial derivative of loss with respect to input data.
        """
        self.dY = dX
        self.dX = self.dY.reshape(self.inputs.shape)
        return self.dX

class Dropout(Layer):
    """
    A class that performs dropout layer operations.

    Attributes
    ----------
    p : float
        Dropout probability.
    mask : ndarray
        Dropout mask.
    training : bool
        Whether the layer is in training mode.

    Methods
    -------
    forward()
        Computes the forward pass of the dropout layer.
    backward(dX : ndarray)
        Computes the backward pass of the dropout layer.
    """
    def __init__(self, p):
        """
        Initializes the Dropout layer.

        Parameters
        ----------
        p : float
            Dropout probability. Must be between 0 and 1.

        Raises
        ------
        ValueError
            If the dropout probability is not between 0 and 1.
        """
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability has to be between 0 and 1")
        self.p = p
        self.mask = None
        self.training = True

    def forward(self):
        """
        Computes the forward pass of the dropout layer.

        Returns
        -------
        Tensor
            The output data.
        """
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, size=self.inputs.shape)
            self.outputs = Tensor(self.inputs.array * self.mask * 1.0 / (1.0 - self.p))
        else:
            self.outputs = self.inputs
        return self.outputs

    def backward(self, dX):
        """
        Computes the backward pass of the dropout layer.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of loss with respect to output data.

        Returns
        -------
        ndarray
            Partial derivative of loss with respect to input data.
        """
        if self.training:
            self.dX = dX * self.mask * 1.0 / (1.0 - self.p)
        else:
            self.dX = dX
        return self.dX

class Linear:
    """
    A class that performs linear layer operations.

    Attributes
    ----------
    weights : ndarray
        Weights of the linear layer.
    biases : ndarray
        Biases of the linear layer.

    Methods
    -------
    forward()
        Computes the forward pass of the linear function.
    backward(lr : float)
        Computes the backward pass of the linear layer.
    """
    def __init__(self, n_in, n_out):
        """
        Initializes the Linear layer.

        Parameters
        ----------
        n_in : int
            Number of input features.
        n_out : int
            Number of output features.
        """
        self.weights = np.random.rand(n_in, n_out)
        self.biases = np.random.rand(1, n_out)
        self.dW = None
        self.dB = None

    def forward(self):
        """
        Computes the forward pass of the linear function.

        Returns
        -------
        ndarray
            The output data.
        """
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, lr):
        """
        Computes the backward pass of the linear function.

        Parameters
        ----------
        lr : float
            Learning rate.
        """
        self.dW = np.sum(-2 * self.inputs * (self.targets - (self.inputs @ self.weights + self.biases))) / len(self.targets - self.outputs)
        self.dB = np.sum(-2 * (self.targets - (self.inputs @ self.weights + self.biases))) / len(self.targets - self.outputs)
        self.weights -= self.dW * lr
        self.biases -= self.dB * lr
        self.dW *= np.zeros(self.dW.shape, dtype=np.float64)
        self.dB *= np.zeros(self.dB.shape, dtype=np.float64)
