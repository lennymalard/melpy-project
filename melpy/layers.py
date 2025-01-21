import numpy as np
from .im2col import *
from .tensor import *

def check_activation(activation):
    if not isinstance(activation, str) and activation is not None:
        raise TypeError("`activation` must be of type str.")
    if activation.lower() not in ("sigmoid", "tanh", "softmax", "tanh", "relu", "leaky_relu"):
        raise ValueError("'activation' must be one of 'sigmoid', 'softmax', 'tanh', 'relu', 'leaky_relu'.'")
    
def initialize_activation(activation, type):
    if activation is None:
        return None
    check_activation(activation)
    if type is Layer:
        return {"sigmoid": Sigmoid(), "softmax": Softmax(), "tanh": Tanh(), "relu": ReLU(), "leaky_relu": LeakyReLU()}[activation.lower()]
    elif type is Function:
        return {"sigmoid": sigmoid, "softmax": softmax, "tanh": tanh, "relu": relu, "leaky_relu": leaky_relu}[activation.lower()]

def check_input_dims(inputs, ndim):
    if inputs.ndim != ndim:
        raise ValueError(f"`inputs` must have {ndim} dimensions.")

def check_tensor(obj, name, none_allowed=False):
    if not isinstance(obj, Tensor) and not isinstance(obj, Operation) and not none_allowed:
        raise TypeError(f"`{name}` must be a Tensor.")

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
        """
        Zeros the gradients of the Tanh activation function.

        Returns
        -------
        None
        """
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
        Input data with the shape format `(batch size, input size)`.
    outputs : Tensor
        Output data with the shape format `(batch size, output size)`.
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
    activation : Activation, optional
        Activation function.

    Methods
    -------
    forward()
        Computes the forward pass of the dense layer.
    backward(dX : ndarray)
        Computes the backward pass of the dense layer.
    """
    def __init__(self, in_features, out_features, weight_initializer="he_uniform", activation=None):
        """
        Initializes the Dense layer.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        weight_initializer : str, optional
            Weight initialization method ('he_normal', 'glorot_normal', 'he_uniform', 'glorot_uniform').
            Default is 'he_uniform'.
        activation : str, optional
            Activation function ('relu', 'leaky_relu', 'sigmoid', 'softmax', 'tanh').

        Raises
        ------
        ValueError
            If the weight initialization method is invalid.
        """
        super().__init__()

        def initialize_weights(weight_init, in_features, out_features):
            """
            Initializes the weights based on the specified method.

            Parameters
            ----------
            weight_init : str
                Weight initialization method.
            in_features : int
                Number of input features.
            out_features : int
                Number of output features.

            Returns
            -------
            Tensor
                Initialized weights.
            """
            if weight_init.lower() == "he_normal":
                weights = Parameter(np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features), requires_grad=True)
            elif weight_init.lower() == "he_uniform":
                limit = np.sqrt(6 / in_features)
                weights = Parameter(np.random.uniform(-limit, limit, (in_features, out_features)), requires_grad=True)
            elif weight_init.lower() == "glorot_normal":
                weights = Parameter(np.random.randn(in_features, out_features) * np.sqrt(2.0 / (in_features + out_features)), requires_grad=True)
            elif weight_init.lower() == "glorot_uniform":
                limit = np.sqrt(6 / (in_features + out_features))
                weights = Parameter(np.random.uniform(-limit, limit, (in_features, out_features)), requires_grad=True)
            else:
                raise ValueError(
                    "`weight_init` must be either 'he_uniform', 'he_normal', 'glorot_uniform' or 'glorot_normal'.")
            return weights

        self.weights = initialize_weights(weight_initializer, in_features, out_features)
        self.biases = Parameter(np.random.rand(1, out_features), requires_grad=True)
        self.dW = np.zeros_like(self.weights.array)
        self.dB = np.zeros_like(self.biases.array)

        self.activation = initialize_activation(activation, Layer)

    def forward(self):
        """
        Computes the forward pass of the dense layer.

        Returns
        -------
        Tensor
            The output data.
        """
        check_tensor(self.inputs, "inputs")
        check_tensor(self.outputs, "outputs", True)

        self.inputs.requires_grad = True
        self.weights.requires_grad = True
        self.biases.requires_grad = True
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
        """
        Zeros the gradients of the Tanh activation function.

        Returns
        -------
        None
        """
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
        check_tensor(self.inputs, "inputs")
        check_tensor(self.outputs, "outputs", True)
        self.inputs.requires_grad = True
        self.outputs = Tensor(np.maximum(0, self.inputs.array), requires_grad=True)
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
        dA[self.inputs.array < 0] = 0.01
        return dA

    def forward(self):
        """
        Computes the Leaky ReLU forward pass.

        Returns
        -------
        Tensor
            The output data.
        """
        check_tensor(self.inputs, "inputs")
        check_tensor(self.outputs, "outputs", True)
        self.inputs.requires_grad = True
        self.outputs = Tensor(np.where(self.inputs.array > 0, self.inputs.array, self.inputs.array * 0.01), requires_grad=True)
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
        check_tensor(self.inputs, "inputs")
        check_tensor(self.outputs, "outputs", True)
        self.inputs.requires_grad = True
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
        """
        Zeros the gradients of the Tanh activation function.

        Returns
        -------
        None
        """
        self.outputs.zero_grad()

class Tanh(Layer, Activation):
    """
    A class that performs Tanh (Hyperbolic Tangent) activation function operations.

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
        Computes the forward pass of the Tanh activation function.
    backward(dX : ndarray)
        Computes the backward pass of the Tanh activation function.
    zero_grad()
        Zeros the gradients of the Tanh activation function.
    """

    def __init__(self):
        """
        Initializes the Tanh activation function.
        """
        super().__init__()

    def forward(self):
        """
        Computes the forward pass of the Tanh activation function.

        Returns
        -------
        Tensor
            The output data.
        """
        check_tensor(self.inputs, "inputs")
        check_tensor(self.outputs, "outputs", True)
        self.inputs.requires_grad = True
        self.outputs = (exp(self.inputs) - exp(-self.inputs)) / (exp(self.inputs) + exp(-self.inputs))
        return self.outputs

    def backward(self, dX):
        """
        Computes the backward pass of the Tanh activation function.

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
        """
        Zeros the gradients of the Tanh activation function.

        Returns
        -------
        None
        """
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
        check_tensor(self.inputs, "inputs")
        check_tensor(self.outputs, "outputs", True)
        self.inputs.requires_grad = True
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
        """
        Zeros the gradients of the Tanh activation function.

        Returns
        -------
        None
        """
        self.outputs.zero_grad()

class Convolution2D(Layer):
    """
    A class that performs Convolution 2D layer operations.

    Attributes
    ----------
    inputs : Tensor
        Input data with the shape format `(batch size, input channels, height, width)`.
    outputs : Tensor
        Output data with the shape format `(batch size, output channels, height, width)`.
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
    activation : Activation, optional
        Activation of the convolution layer.

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
        activation : str, optional
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
            if not isinstance(padding, str):
                raise TypeError("`padding` must be a string.")

            if padding not in ("valid", "same"):
                raise ValueError("`padding` must be 'valid' or 'same'.")

            if weight_init.lower() == "he_normal":
                weights = Parameter(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(
                    2.0 / (in_channels * kernel_size ** 2)), requires_grad=True)
            elif weight_init.lower() == "he_uniform":
                limit = np.sqrt(6 / (in_channels * kernel_size ** 2))
                weights = Parameter(
                    np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size)), requires_grad=True)
            elif weight_init.lower() == "glorot_normal":
                weights = Parameter(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(
                    2.0 / (in_channels * kernel_size ** 2 + out_channels * kernel_size ** 2)), requires_grad=True)
            elif weight_init.lower() == "glorot_uniform":
                limit = np.sqrt(6 / (in_channels * kernel_size ** 2 + out_channels * kernel_size ** 2))
                weights = Parameter(
                    np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size)), requires_grad=True)
            else:
                raise ValueError(
                    "`weight_init` must be either 'he_uniform', 'he_normal', 'glorot_uniform' or 'glorot_normal'.")
            return weights

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = initialize_weights(weight_initializer, in_channels, out_channels, kernel_size)
        self.dW = np.zeros_like(self.weights)
        if use_bias is True:
            self.biases = Parameter(np.zeros(shape=(1, out_channels, 1, 1)).astype(np.float64), requires_grad=True)
            self.dB = np.zeros_like(self.biases)
        else:
            self.biases = None
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride

        self.activation = initialize_activation(activation, Layer)

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
        return Tensor(np.pad(self.inputs.array, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant'), requires_grad=True)

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
        check_tensor(self.inputs, "inputs")
        check_tensor(self.outputs, "outputs", True)
        check_input_dims(self.inputs, 4)
        self.inputs.requires_grad = True
        self.weights.requires_grad = True
        self.biases.requires_grad = True
        self.input_padded = self.explicit_padding()

        self.input_cols = im2col(self.input_padded, self.kernel_size, self.stride)
        self.filter_cols = Tensor(self.weights.array.reshape(self.out_channels, -1), requires_grad=True)

        output_height, output_width = self.get_output_size(self.inputs.shape[2], self.inputs.shape[3])

        self.output_cols = self.filter_cols @ self.input_cols

        self.outputs = Tensor(np.array(np.hsplit(self.output_cols.array, self.inputs.shape[0])).reshape(
            (self.input_padded.shape[0], self.out_channels, output_height, output_width)
        ), requires_grad=True)

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

        self.dX_padded = col2im(Tensor(self.dX_cols, requires_grad=True), self.input_padded.shape, self.kernel_size, self.stride)

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
        """
        Zeros the gradients of the Tanh activation function.

        Returns
        -------
        None
        """
        if self.activation is not None:
            self.activation.zero_grad()
        self.output_cols.zero_grad()

class Pooling2D(Layer):
    """
    A class that performs 2D pooling layer operations.

    Attributes
    ----------
    inputs : Tensor
        Input data with the shape format `(batch size, channels, height, width)`.
    outputs : Tensor
        Output data with the shape format `(batch size, channels, height, width)`.
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
        check_tensor(self.inputs, "inputs")
        check_tensor(self.outputs, "outputs", True)
        check_input_dims(self.inputs, 4)

        self.inputs.requires_grad = True
        output_height = int((self.inputs.shape[2] - self.pool_size + self.stride) // self.stride)
        output_width = int((self.inputs.shape[3] - self.pool_size + self.stride) // self.stride)

        output_shape = (self.inputs.shape[0], self.inputs.shape[1], output_height, output_width)

        self.input_cols = im2col(self.inputs, self.pool_size, self.stride)
        self.input_cols_reshaped = Tensor(
            np.array(np.hsplit(np.array(np.hsplit(self.input_cols.array, self.inputs.shape[0])), self.inputs.shape[1])), requires_grad=True)

        self.maxima = max(self.input_cols_reshaped, axis=2)
        self.maxima_reshaped = self.maxima.array.reshape(self.inputs.shape[1], -1)

        self.outputs = col2im(Tensor(self.maxima_reshaped, requires_grad=True), output_shape, 1, 1)

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

        self.dY_cols = im2col(Tensor(self.dY, requires_grad=True), 1, 1)
        self.dY_cols_reshaped = np.array(
            np.hsplit(np.array(np.hsplit(self.dY_cols.array, self.dY.shape[0])), self.dY.shape[1]))

        self.maxima.backward(self.dY_cols_reshaped)

        self.dX_cols = np.concatenate(np.concatenate(self.input_cols_reshaped.grad, axis=1), axis=1)

        self.dX = col2im(Tensor(self.dX_cols, requires_grad=True), self.inputs.shape, self.pool_size, self.stride).array

        return self.dX

    def zero_grad(self):
        """
        Zeros the gradients of the Tanh activation function.

        Returns
        -------
        None
        """
        self.maxima.zero_grad()

class Flatten(Layer):
    """
    A class that performs flattening layer operations.

    Attributes
    ----------
    inputs : Tensor
        Input data.
    outputs : Tensor
        Output data.
    dX : ndarray
        Partial derivative of loss with respect to input data.
    dY : ndarray
        Partial derivative of loss with respect to output data.

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
        check_tensor(self.inputs, "inputs")
        check_tensor(self.outputs, "outputs", True)
        self.inputs.requires_grad = True
        self.outputs = Tensor(self.inputs.array.reshape((self.inputs.shape[0], -1)), requires_grad=True)
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
    inputs : Tensor
        Input data.
    outputs : Tensor
        Output data.
    dX : ndarray
        Partial derivative of loss with respect to input data.
    dY : ndarray
        Partial derivative of loss with respect to output data.
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
        check_tensor(self.inputs, "inputs")
        check_tensor(self.outputs, "outputs", True)
        self.inputs.requires_grad = True
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, size=self.inputs.shape)
            self.outputs = Tensor(self.inputs.array * self.mask * 1.0 / (1.0 - self.p), requires_grad=True)
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

class LSTMCell(Layer):
    """
    Implements a single Long Short-Term Memory (LSTM) cell.

    Attributes
    ----------
    inputs : Tensor
        Input data with the shape format `(batch size, input size)`.
    outputs : Tensor
        Output data with the shape format `(batch size, hidden size)`.
    input_size : int
        Number of input features.
    hidden_size : int
        Number of hidden units.
    activation : Function
        Activation function for the cell state and output.
    use_bias : bool
        Whether to use biases in the cell.
    input_weights, hidden_weights : Weights
        Weight matrices for input and hidden states, respectively.
    biases : Biases, optional
        Bias terms for input, forget, cell, and output gates.
    cell_state : Tensor
        Current cell state.
    hidden_state : Tensor
        Current hidden state.
    input_gate, forget_gate, output_gate, cell_input : Tensor
        Intermediate gate computations.
    dH : ndarray
            Gradient of loss with respect to the previous hidden state.
    dC : ndarray
        Gradient of loss with respect to the previous cell state.
    dX : ndarray
        Gradient of loss with respect to the input.

    Methods
    -------
    forward(hidden_state, cell_state)
        Computes the forward pass, updating hidden and cell states.
    backward(hidden_state_grad, cell_state_grad)
        Computes gradients for inputs and parameters.
    zero_grad()
        Resets all gradients to zero.
    clear_memory()
        Clears stored intermediate states for a new sequence.
    """

    def __init__(self, input_size, hidden_size, activation="tanh", use_bias=True):
        """
        Initialize the LSTM cell.

        Parameters
        ----------
        input_size : int
            Number of input features.
        hidden_size : int
            Number of hidden units.
        activation : str, optional
            Activation function for cell state/output. Default is 'tanh'.
        use_bias : bool, optional
            Whether to include biases. Default is True.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = initialize_activation(activation, Function)
        self.use_bias = use_bias

        self.input_gate = None
        self.output_gate = None
        self.forget_gate = None
        self.next_forget_gate = None
        self.cell_input = None
        self.cell_state = None
        self.previous_cell_state = None
        self.hidden_state = None
        self.previous_hidden_state = None

        self.sequence_hidden_states = []
        self.sequence_cell_states = []
        self.sequence_input_gates= []
        self.sequence_forget_gates = []
        self.sequence_cell_inputs = []
        self.sequence_output_gates = []
        self.sequence_inputs = []
        self.sequence_inputs_grads = []

        self.input_weights = self.Weights((input_size, hidden_size))
        self.hidden_weights = self.Weights((hidden_size, hidden_size))
        self.biases = self.Biases((1, hidden_size)) if use_bias else None

        self.i_input_weights, self.f_input_weights, self.c_input_weights, self.o_input_weights = self.input_weights.split()
        self.i_hidden_weights, self.f_hidden_weights, self.c_hidden_weights, self.o_hidden_weights = self.hidden_weights.split()
        self.i_biases, self.f_biases, self.c_biases, self.o_biases = self.biases.split()

        self.dC = None
        self.dH = None

    def forward(self, inputs, hidden_state, cell_state):
        """
        Perform the forward pass of the LSTM cell.

        Parameters
        ----------
        hidden_state : Tensor
            Previous hidden state.
        cell_state : Tensor
            Previous cell state.

        Returns
        -------
        hidden_state : Tensor
            Updated hidden state.
        cell_state : Tensor
            Updated cell state.
        """
        check_tensor(inputs, "inputs")
        check_tensor(self.outputs, "outputs", True)
        check_input_dims(inputs, 2)

        self.inputs = inputs
        self.previous_hidden_state = Tensor(hidden_state.array)
        self.previous_cell_state = Tensor(cell_state.array)

        input_gate = self.inputs @ self.i_input_weights + self.previous_hidden_state @ self.i_hidden_weights
        output_gate = self.inputs @ self.o_input_weights + self.previous_hidden_state @ self.o_hidden_weights
        forget_gate = self.inputs @ self.f_input_weights + self.previous_hidden_state @ self.f_hidden_weights
        cell_input = self.inputs @ self.c_input_weights + self.previous_hidden_state @ self.c_hidden_weights

        if self.use_bias:
            input_gate += self.i_biases
            output_gate += self.o_biases
            forget_gate += self.f_biases
            cell_input += self.c_biases

        self.input_gate = sigmoid(input_gate)
        self.output_gate = sigmoid(output_gate)
        self.forget_gate = sigmoid(forget_gate)
        self.cell_input = self.activation(cell_input)

        self.cell_state = self.forget_gate * self.previous_cell_state + self.input_gate * self.cell_input
        self.hidden_state = self.output_gate * self.activation(self.cell_state)
        self.outputs = self.hidden_state

        self.sequence_hidden_states.append(self.hidden_state)
        self.sequence_cell_states.append(self.cell_state)
        self.sequence_input_gates.append(self.input_gate)
        self.sequence_forget_gates.append(self.forget_gate)
        self.sequence_cell_inputs.append(self.cell_input)
        self.sequence_output_gates.append(self.output_gate)
        self.sequence_inputs.append(self.inputs)

        return self.hidden_state, self.cell_state

    def backward(self, hidden_state_grad, cell_state_grad):
        """
        Perform the backward pass of the LSTM cell.

        Parameters
        ----------
        hidden_state_grad : ndarray
            Gradient of loss with respect to the hidden state.
        cell_state_grad : ndarray
            Gradient of loss with respect to the cell state.

        Returns
        -------
        dH : ndarray
            Gradient of loss with respect to the previous hidden state.
        dC : ndarray
            Gradient of loss with respect to the previous cell state.
        dX : ndarray
            Gradient of loss with respect to the input.
        """
        self.dC = (Tensor(hidden_state_grad) * self.output_gate * Tensor(self.activation(self.cell_state).derivative()) +
                   Tensor(cell_state_grad) * self.next_forget_gate).array

        dI = self.dC * self.cell_input.array
        dF = self.dC * self.previous_cell_state.array
        dO = hidden_state_grad * self.activation(self.cell_state).array
        dTildeC = self.dC * self.input_gate.array

        i_dZ = dI * self.input_gate.array * (1 - self.input_gate.array)
        f_dZ = dF * self.forget_gate.array * (1 - self.forget_gate.array)
        o_dZ = dO * self.output_gate.array * (1 - self.output_gate.array)
        c_dZ = dTildeC * self.cell_input.derivative()

        self.i_input_weights.grad += self.inputs.T.array @ i_dZ
        self.f_input_weights.grad += self.inputs.T.array @ f_dZ
        self.o_input_weights.grad += self.inputs.T.array @ o_dZ
        self.c_input_weights.grad += self.inputs.T.array @ c_dZ

        self.i_hidden_weights.grad += self.previous_hidden_state.T.array @ i_dZ
        self.f_hidden_weights.grad += self.previous_hidden_state.T.array @ f_dZ
        self.o_hidden_weights.grad += self.previous_hidden_state.T.array @ o_dZ
        self.c_hidden_weights.grad += self.previous_hidden_state.T.array @ c_dZ

        if self.use_bias:
            self.i_biases.grad += np.sum(i_dZ, axis=0, keepdims=True)
            self.f_biases.grad += np.sum(f_dZ, axis=0, keepdims=True)
            self.o_biases.grad += np.sum(o_dZ, axis=0, keepdims=True)
            self.c_biases.grad += np.sum(c_dZ, axis=0, keepdims=True)

        self.dH = i_dZ @ self.i_hidden_weights.T.array + f_dZ @ self.f_hidden_weights.T.array + \
                  o_dZ @ self.o_hidden_weights.T.array + c_dZ @ self.c_hidden_weights.T.array

        self.dX = i_dZ @ self.i_input_weights.T.array + f_dZ @ self.f_input_weights.T.array + \
                  o_dZ @ self.o_input_weights.T.array + c_dZ @ self.c_input_weights.T.array

        self.sequence_inputs_grads.append(self.dX)

        return  self.dX, self.dH, self.dC

    def clear_memory(self):
        """
        Clear stored intermediate states for the cell.
        """
        self.sequence_hidden_states = []
        self.sequence_cell_states = []
        self.sequence_input_gates= []
        self.sequence_forget_gates = []
        self.sequence_cell_inputs = []
        self.sequence_output_gates = []
        self.sequence_inputs = []

    def zero_grad(self):
        """
        Zeros the gradients of the LSTM cell.

        Returns
        -------
        None
        """
        self.i_input_weights.zero_grad()
        self.f_input_weights.zero_grad()
        self.c_input_weights.zero_grad()
        self.o_input_weights.zero_grad()

        self.i_hidden_weights.zero_grad()
        self.f_hidden_weights.zero_grad()
        self.c_hidden_weights.zero_grad()
        self.o_hidden_weights.zero_grad()

        self.i_biases.zero_grad()
        self.f_biases.zero_grad()
        self.o_biases.zero_grad()
        self.c_biases.zero_grad()

        self.sequence_inputs_grads = []

    class Weights(Parameter):
        """
        A class that initializes the weights of the LSTM cell.

        Methods
        -------
        split()
            Splits the tensor in 4 parameters of type Parameter.
        """
        def __init__(self, shape):
            limit = np.sqrt(6 / (shape[0] + 4 * shape[1]))
            array = np.random.uniform(-limit, limit, (shape[0], 4 * shape[1]))
            super().__init__(object=array)

        def split(self):
            # Grad not required because it would break direct backward computations
            return tuple(Parameter(W) for W in np.hsplit(self.array, 4))

    class Biases(Parameter):
        """

        A class that initializes the biases of the LSTM cell.

        Methods
        -------
        split()
            Splits the tensor in 4 parameters of type Parameter.
        """
        def __init__(self, shape):
            if shape[0] != 1:
                raise ValueError("The shape at index 0 must be 1.")
            limit = np.sqrt(6 / (shape[0] + 4 * shape[1]))
            array = np.random.uniform(-limit, limit, (shape[0], 4 * shape[1]))
            super().__init__(object=array)

        def split(self):
            # Grad not required because it would break direct backward computations
            return tuple(Parameter(W.reshape(1,-1)) for W in np.hsplit(self.array, 4))

class LSTM(Layer):
    """
    Implements a Long Short-Term Memory (LSTM) layer for sequential data.

    Attributes
    ----------
    inputs : Tensor
        Input data with the shape format `(batch size, sequence length, input size)`.
    outputs : Tensor
        Output data with the shape format `(batch size, sequence length, hidden size)`.
    input_size : int
        Number of input features.
    hidden_size : int
        Number of hidden units in each cell.
    num_layers : int
        Number of stacked LSTM layers.
    use_bias : bool
        Whether to include biases in the cells.
    cells : list of LSTMCell
        List of LSTMCell instances for each layer.
    return_sequences : bool
        Whether to return the full sequence of hidden states.

    Methods
    -------
    forward()
        Performs the forward pass for a sequence of inputs.
    backward(dX)
        Computes gradients for inputs and parameters.
    zero_grad()
        Resets gradients for all LSTM cells.
    """

    def __init__(self, input_size, hidden_size, activation="tanh", num_layers=1, use_bias=True, return_sequences=False):
        """
        Initialize the LSTM layer.

        Parameters
        ----------
        input_size : int
            Number of input features.
        hidden_size : int
            Number of hidden units in each cell.
        activation : str, optional
            Activation function for the cells. Default is 'tanh'.
        num_layers : int, optional
            Number of stacked LSTM layers. Default is 1.
        use_bias : bool, optional
            Whether to include biases. Default is True.
        return_sequences : bool, optional
             Whether to return the full sequence of hidden states.
        """
        check_activation(activation)
        
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cells = [
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size, activation, use_bias=use_bias)
            for i in range(num_layers)
        ]

    def sequence_to_tensor(self, sequence):
        if not isinstance(sequence, list):
            raise TypeError("'sequence' must be a list.")
        return Tensor(np.stack([e.array if (isinstance(e, Tensor) or isinstance(e, Operation)) else e for e in sequence], axis=1))

    def forward(self):
        """
        Performs the forward pass through the LSTM layer.

        For each timestep in the sequence, computes the hidden and cell states for each LSTM cell in the layer.

        Returns
        -------
        outputs : Tensor
            The hidden state output of the last LSTM cell in the last layer.
        """
        check_tensor(self.inputs, "inputs")
        check_tensor(self.outputs, "outputs", True)
        check_input_dims(self.inputs, 3)
        batch_size, sequence_length = self.inputs.shape[:2]

        for cell in self.cells:
            cell.clear_memory()

        inputs = self.inputs

        for i, cell in enumerate(self.cells):
            inputs = self.sequence_to_tensor(self.cells[i-1].sequence_hidden_states) if i > 0 else inputs
            for t in range(sequence_length):
                input_sequence = Tensor(inputs.array[:, t])
                hidden_state = zeros((batch_size, self.hidden_size)) if t == 0 else cell.sequence_hidden_states[t-1]
                cell_state = zeros((batch_size, self.hidden_size)) if t == 0 else cell.sequence_cell_states[t-1]
                hidden_state, cell_state = cell.forward(input_sequence, hidden_state, cell_state)

        self.outputs = hidden_state

        if self.return_sequences:
            self.outputs = self.sequence_to_tensor(self.cells[-1].sequence_hidden_states)

        return self.outputs

    def backward(self, dX):
        """
        Performs the backward pass through the LSTM layer.

        Computes gradients for all inputs, weights, biases, and hidden states.

        Parameters
        ----------
        dX : ndarray
            Gradient of loss with respect to the output.

        Returns
        -------
        dX : ndarray
            Gradient of loss with respect to the input.
        """
        batch_size, sequence_length, input_size = self.inputs.shape
        self.dY = dX
        hidden_state_grad = dX
        for i in reversed(range(len(self.cells))):
            if i < len(self.cells) - 1 or self.return_sequences:
                hidden_state_grad = np.zeros_like(self.cells[i].hidden_state.array)
            cell_state_grad = np.zeros((batch_size, self.hidden_size))

            for t in reversed(range(sequence_length)):
                if t < sequence_length-1:
                    self.cells[i].next_forget_gate = self.cells[i].sequence_forget_gates[t+1]
                else:
                    self.cells[i].next_forget_gate = zeros_like(self.cells[i].forget_gate)

                self.cells[i].input_gate = self.cells[i].sequence_input_gates[t]
                self.cells[i].output_gate = self.cells[i].sequence_output_gates[t]
                self.cells[i].forget_gate = self.cells[i].sequence_forget_gates[t]
                self.cells[i].cell_input = self.cells[i].sequence_cell_inputs[t]
                self.cells[i].hidden_state = self.cells[i].sequence_hidden_states[t]
                self.cells[i].cell_state = self.cells[i].sequence_cell_states[t]
                self.cells[i].inputs = self.cells[i].sequence_inputs[t]

                if t > 0:
                    self.cells[i].previous_hidden_state = self.cells[i].sequence_hidden_states[t-1]
                    self.cells[i].previous_cell_state = self.cells[i].sequence_cell_states[t-1]
                else:
                    self.cells[i].previous_hidden_state = zeros((batch_size, self.hidden_size))
                    self.cells[i].previous_cell_state = zeros((batch_size, self.hidden_size))

                if i < len(self.cells) - 1:
                    hidden_state_grad += self.sequence_to_tensor(
                        list(reversed(self.cells[i + 1].sequence_inputs_grads))).array[:, t, :]

                elif i == len(self.cells) - 1 and self.return_sequences:
                    hidden_state_grad += self.dY[:, t, :]

                self.cells[i].sequence_hidden_states[t].grad = hidden_state_grad
                self.cells[i].sequence_cell_states[t].grad = cell_state_grad

                dX, hidden_state_grad, cell_state_grad = self.cells[i].backward(hidden_state_grad, cell_state_grad)

                self.cells[i].sequence_inputs[t].grad = dX

        self.dX = self.sequence_to_tensor(list(reversed(self.cells[0].sequence_inputs_grads)))

        self.inputs.grad = self.dX
        self.outputs.grad = self.dY

        return self.dX

    def zero_grad(self):
        """
        Resets the gradients of all LSTM cells in the layer to zero.

        Returns
        -------
        None
        """
        for cell in self.cells:
            cell.zero_grad()