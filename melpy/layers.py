import numpy as np
from .im2col import *
from .Tensor import *

def check_activation(activation):
    if not isinstance(activation, str) and activation is not None:
        raise TypeError("`activation` must be of type str.")
    if activation is not None and activation.lower() not in ("sigmoid", "tanh", "softmax", "tanh", "relu", "leaky_relu"):
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

def check_ohe(obj, name):
    if not np.unique(obj.array).all() == np.array([0, 1]).all():
        raise ValueError(f"`{name}` must be one-hot encoded.")

def flatten_tensor(tensor):
    return tensor.reshape(-1, tensor.shape[-1])

def restore_flattened(tensor, shape):
    return tensor.reshape(shape)

class Layer:
    """
    Base class for all layers in the neural network.

    Attributes
    ----------
    inputs : Tensor
        Input data.
    outputs : Tensor
        Output data.
    dX : Tensor
        Partial derivative of loss with respect to input.
    dY : Tensor
        Partial derivative of loss with respect to output.

    Methods
    -------
    derivative()
        Computes the derivative of the layer.
    forward()
        Computes the forward pass of the layer.
    backward(dX : Tensor)
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
        dX : Tensor
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
    dX : Tensor
        Partial derivative of loss with respect to input.
    dY : Tensor
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
    backward(dX : Tensor)
        Computes the backward pass of the dense layer.
    """
    def __init__(self, in_features, out_features, activation=None, weight_initializer="he_uniform"):
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

        self.parameters = [initialize_weights(weight_initializer, in_features, out_features), Parameter(np.random.rand(1, out_features), requires_grad=True)]

        self.dW = zeros_like(self.weights)
        self.dB = zeros_like(self.biases)

        self.in_features = in_features
        self.out_features = out_features
        self.leading_dims = ()

        self.flattened = False

        self.activation = initialize_activation(activation, Layer)

    @property
    def weights(self):
        return self.parameters[0]

    @weights.setter
    def weights(self, value):
        self.parameters[0] = value

    @property
    def biases(self):
        return self.parameters[1]

    @biases.setter
    def biases(self, value):
        self.parameters[1] = value

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

        if self.inputs.ndim > 2:
            self.leading_dims = self.inputs.shape[:-1]
            self.flattened = True
            self.inputs = flatten_tensor(self.inputs)

        self.inputs.requires_grad = True
        self.weights.requires_grad = True
        self.biases.requires_grad = True
        self.outputs = dot(self.inputs, self.weights) + self.biases

        if self.activation is not None:
            self.activation.inputs = self.outputs
            return self.activation.forward()

        if self.flattened:
            input_shape = (*self.leading_dims, self.in_features)
            output_shape = (*self.leading_dims, self.out_features)
            self.inputs = restore_flattened(self.inputs, input_shape)
            self.outputs = restore_flattened(self.outputs, output_shape)

        return self.outputs

    def backward(self, dX):
        """
        Computes the backward pass of the dense layer.

        Parameters
        ----------
        dX : Tensor
            Partial derivative of loss with respect to output data.

        Returns
        -------
        ndarray
            Partial derivative of loss with respect to input data.
        """
        if self.activation is not None:
            dX = self.activation.backward(dX)

        self.dY = dX

        if self.flattened:
            dX = flatten_tensor(dX)
            self.inputs = flatten_tensor(self.inputs)
            self.outputs = flatten_tensor(self.outputs)

        self.outputs.backward(dX)
        self.dX = self.inputs.grad
        self.dW = self.weights.grad

        if self.flattened:
            dX_shape = (*self.leading_dims, self.in_features)
            input_shape = (*self.leading_dims, self.in_features)
            output_shape = (*self.leading_dims, self.out_features)
            self.dX = restore_flattened(self.dX, dX_shape)
            self.inputs = restore_flattened(self.inputs, input_shape)
            self.outputs = restore_flattened(self.outputs, output_shape)

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
    dX : Tensor
        Partial derivative of loss with respect to input.
    dY : Tensor
        Partial derivative of loss with respect to output.

    Methods
    -------
    derivative()
        Computes the ReLU derivative.
    forward()
        Computes the ReLU forward pass.
    backward(dX : Tensor)
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
        return Tensor(dA)

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
        dX : Tensor
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
    dX : Tensor
        Partial derivative of loss with respect to input.
    dY : Tensor
        Partial derivative of loss with respect to output.

    Methods
    -------
    derivative()
        Computes the Leaky ReLU derivative.
    forward()
        Computes the Leaky ReLU forward pass.
    backward(dX : Tensor)
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
        return Tensor(dA)

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
        dX : Tensor
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
    dX : Tensor
        Partial derivative of loss with respect to input.
    dY : Tensor
        Partial derivative of loss with respect to output.

    Methods
    -------
    derivative()
        Computes the Sigmoid derivative.
    forward()
        Computes the Sigmoid forward pass.
    backward(dX : Tensor)
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
        dX : Tensor
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
    dX : Tensor
        Partial derivative of loss with respect to input.
    dY : Tensor
        Partial derivative of loss with respect to output.

    Methods
    -------
    forward()
        Computes the forward pass of the Tanh activation function.
    backward(dX : Tensor)
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
        dX : Tensor
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
    dX : Tensor
        Partial derivative of loss with respect to input.
    dY : Tensor
        Partial derivative of loss with respect to output.

    Methods
    -------
    forward()
        Computes the Softmax forward pass.
    backward(dX : Tensor)
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
        dX : Tensor
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
    backward(dX : Tensor)
        Computes the backward pass of the convolution layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding="valid", stride=1, weight_initializer="he_uniform", use_bias=True):
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
        self.parameters = [initialize_weights(weight_initializer, in_channels, out_channels, kernel_size), None]
        self.dW = zeros_like(self.weights)

        if use_bias is True:
            self.parameters[1] = Parameter(np.zeros(shape=(1, out_channels, 1, 1)).astype(np.float64), requires_grad=True)
            self.dB = zeros_like(self.biases)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_bias = use_bias

        self.activation = initialize_activation(activation, Layer)

        if self.padding == "same":
            self.stride = 1

    @property
    def weights(self):
        return self.parameters[0]

    @weights.setter
    def weights(self, value):
        self.parameters[0] = value

    @property
    def biases(self):
        return self.parameters[1]

    @biases.setter
    def biases(self, value):
        self.parameters[1] = value

    def calculate_padding(self):
        # TODO Needs improvement
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
        return None 

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
        output_height, output_width = 0, 0
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

        if self.use_bias:
            self.outputs += self.biases

        if self.activation is not None:
            self.activation.inputs = self.outputs
            return self.activation.forward()

        return self.outputs

    def backward(self, dX):
        """
        Computes the backward pass of the convolution layer.

        Parameters
        ----------
        dX : Tensor
            Partial derivative of loss with respect to output data.

        Returns
        -------
        ndarray
            Partial derivative of loss with respect to input data.
        """
        if self.activation is not None:
            dX = self.activation.backward(dX)

        self.dY = dX

        self.dY_reshaped = self.dY.array.reshape(self.dY.shape[0] * self.dY.shape[1], self.dY.shape[2] * self.dY.shape[3])
        self.dY_reshaped = np.vsplit(self.dY_reshaped, self.inputs.shape[0])
        self.dY_reshaped = np.concatenate(self.dY_reshaped, axis=-1)

        self.output_cols.backward(Tensor(self.dY_reshaped, requires_grad=True))

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
    backward(dX : Tensor)
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
        dX : Tensor
            Partial derivative of loss with respect to output data.

        Returns
        -------
        ndarray
            Partial derivative of loss with respect to input data.
        """
        self.dY = dX
        self.dX = zeros_like(self.inputs)

        self.dY_cols = im2col(Tensor(self.dY, requires_grad=True), 1, 1)
        self.dY_cols_reshaped = np.hsplit(np.array(np.hsplit(self.dY_cols.array, self.dY.shape[0])), self.dY.shape[1])

        self.maxima.backward(Tensor(self.dY_cols_reshaped, requires_grad=True))

        self.dX_cols = np.concatenate(np.concatenate(self.input_cols_reshaped.grad.array, axis=1), axis=1)

        self.dX = col2im(Tensor(self.dX_cols, requires_grad=True), self.inputs.shape, self.pool_size, self.stride)

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
    dX : Tensor
        Partial derivative of loss with respect to input data.
    dY : Tensor
        Partial derivative of loss with respect to output data.

    Methods
    -------
    forward()
        Computes the forward pass of the flattening layer.
    backward(dX : Tensor)
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
        dX : Tensor
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
    dX : Tensor
        Partial derivative of loss with respect to input data.
    dY : Tensor
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
    backward(dX : Tensor)
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
        dX : Tensor
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
    dH : Tensor
            Gradient of loss with respect to the previous hidden state.
    dC : Tensor
        Gradient of loss with respect to the previous cell state.
    dX : Tensor
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

        self.parameters = [
            self.Weights((input_size, hidden_size)),
            self.Weights((hidden_size, hidden_size))
        ]

        if self.use_bias:
            self.parameters.append(self.Biases((1, hidden_size)))

        self.dC = None
        self.dH = None

    @property
    def input_weights(self):
        return self.parameters[0]

    @input_weights.setter
    def input_weights(self, value):
        self.parameters[0] = value

    @property
    def hidden_weights(self):
        return self.parameters[1]

    @hidden_weights.setter
    def hidden_weights(self, value):
        self.parameters[1] = value

    @property
    def biases(self):
        if self.use_bias:
            return self.parameters[2]
        return None

    @biases.setter
    def biases(self, value):
        if self.use_bias:
            self.parameters[2] = value

    def forward(self, precomputed_input_part, hidden_state, cell_state):
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
        check_tensor(precomputed_input_part, "inputs")
        check_tensor(self.outputs, "outputs", True)
        check_input_dims(precomputed_input_part, 2)

        self.previous_hidden_state = Tensor(hidden_state.array)
        self.previous_cell_state = Tensor(cell_state.array)

        gates = precomputed_input_part + self.previous_hidden_state @ self.hidden_weights

        if self.use_bias:
            gates += self.biases

        input_gate, forget_gate, cell_input, output_gate = np.hsplit(gates.array, 4)

        self.input_gate = sigmoid(Tensor(input_gate))
        self.forget_gate = sigmoid(Tensor(forget_gate))
        self.cell_input = (self.activation(Tensor(cell_input)), self.activation(Tensor(cell_input), derivative=True))
        self.output_gate = sigmoid(Tensor(output_gate))

        self.cell_state = self.forget_gate * self.previous_cell_state + self.input_gate * self.cell_input[0]
        self.hidden_state = self.output_gate * self.activation(self.cell_state)
        self.outputs = self.hidden_state

        self.sequence_hidden_states.append(self.hidden_state)
        self.sequence_cell_states.append(self.cell_state)
        self.sequence_input_gates.append(self.input_gate)
        self.sequence_forget_gates.append(self.forget_gate)
        self.sequence_cell_inputs.append(self.cell_input)
        self.sequence_output_gates.append(self.output_gate)

        return self.hidden_state, self.cell_state

    def backward(self, hidden_state_grad, cell_state_grad):
        """
        Perform the backward pass of the LSTM cell.

        Parameters
        ----------
        hidden_state_grad : Tensor
            Gradient of loss with respect to the hidden state.
        cell_state_grad : Tensor
            Gradient of loss with respect to the cell state.

        Returns
        -------
        dH : Tensor
            Gradient of loss with respect to the previous hidden state.
        dC : Tensor
            Gradient of loss with respect to the previous cell state.
        dX : Tensor
            Gradient of loss with respect to the input.
        """
        self.dC = (hidden_state_grad * self.output_gate * self.activation(self.cell_state, derivative=True) +
                   cell_state_grad * self.next_forget_gate)

        dI = self.dC * self.cell_input[0]
        dF = self.dC * self.previous_cell_state
        dO = hidden_state_grad * self.activation(self.cell_state)
        dTildeC = self.dC * self.input_gate

        gates_grad = np.hstack(
            [dI.array * self.input_gate.array * (1 - self.input_gate.array),
             dF.array * self.forget_gate.array * (1 - self.forget_gate.array),
             dTildeC.array * self.cell_input[1].array,
             dO.array * self.output_gate.array * (1 - self.output_gate.array)
             ]
        )
        gates_grad = Tensor(gates_grad)

        self.input_weights.grad += self.inputs.T @ gates_grad
        self.hidden_weights.grad += self.previous_hidden_state.T @ gates_grad

        if self.use_bias:
            self.biases.grad += sum(gates_grad, axis=0, keepdims=True)

        self.dH = (gates_grad.array[:, :self.hidden_size] @ self.hidden_weights.array[:, :self.hidden_size].T +
                   gates_grad.array[:, self.hidden_size:2*self.hidden_size] @ self.hidden_weights.array[:, self.hidden_size:2*self.hidden_size].T +
                   gates_grad.array[:, 2*self.hidden_size:3*self.hidden_size] @ self.hidden_weights.array[:, 2*self.hidden_size:3*self.hidden_size].T +
                   gates_grad.array[:, 3*self.hidden_size:4*self.hidden_size] @ self.hidden_weights.array[:, 3*self.hidden_size:4*self.hidden_size].T)

        self.dH = Tensor(self.dH)

        self.dX = (gates_grad.array[:, :self.hidden_size] @ self.input_weights.array[:, :self.hidden_size].T +
                   gates_grad.array[:, self.hidden_size:2*self.hidden_size] @ self.input_weights.array[:, self.hidden_size:2*self.hidden_size].T +
                   gates_grad.array[:, 2*self.hidden_size:3*self.hidden_size] @ self.input_weights.array[:, 2*self.hidden_size:3*self.hidden_size].T +
                   gates_grad.array[:, 3*self.hidden_size:4*self.hidden_size] @ self.input_weights.array[:, 3*self.hidden_size:4*self.hidden_size].T)

        self.dX = Tensor(self.dX)

        self.sequence_inputs_grads.append(self.dX)

        return self.dX, self.dH, self.dC

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
        self.input_weights.zero_grad()
        self.hidden_weights.zero_grad()

        if self.use_bias:
            self.biases.zero_grad()

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
            precomputed_input_part = inputs @ cell.input_weights
            for t in range(sequence_length):
                cell.sequence_inputs.append(Tensor(inputs.array[:, t]))
                hidden_state = zeros((batch_size, self.hidden_size)) if t == 0 else cell.sequence_hidden_states[t-1]
                cell_state = zeros((batch_size, self.hidden_size)) if t == 0 else cell.sequence_cell_states[t-1]
                hidden_state, cell_state = cell.forward(Tensor(precomputed_input_part.array[:, t]), hidden_state, cell_state)

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
        dX : Tensor
            Gradient of loss with respect to the output.

        Returns
        -------
        dX : Tensor
            Gradient of loss with respect to the input.
        """
        batch_size, sequence_length, input_size = self.inputs.shape
        self.dY = dX
        hidden_state_grad = dX
        for i in reversed(range(len(self.cells))):
            if i < len(self.cells) - 1 or self.return_sequences:
                hidden_state_grad = zeros_like(self.cells[i].hidden_state)
            cell_state_grad = zeros((batch_size, self.hidden_size))

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
                    hidden_state_grad += Tensor(self.sequence_to_tensor(
                        list(reversed(self.cells[i + 1].sequence_inputs_grads))).array[:, t, :])

                elif i == len(self.cells) - 1 and self.return_sequences:
                    hidden_state_grad += Tensor(self.dY.array[:, t, :])

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

class Embedding(Layer):
    """
    A layer that maps positive integers (indices) into dense vectors of fixed size.

    This layer converts one-hot encoded vectors into dense embeddings via a learned
    weight matrix. The forward pass computes a matrix multiplication between the
    one-hot inputs and the embedding weights, effectively performing an embedding lookup.

    Parameters
    ----------
    input_dim : int
       Size of the vocabulary.
    output_dim : int
       Dimension of the dense embedding (size of each embedding vector).
    weight_initializer : str, optional (default="he_uniform")
       Weight initialization strategy. One of:
       - "he_uniform": Uniform distribution scaled by sqrt(6 / input_dim)
       - "he_normal": Normal distribution scaled by sqrt(2 / input_dim)
       - "glorot_uniform": Uniform distribution scaled by sqrt(6 / (input_dim + output_dim))
       - "glorot_normal": Normal distribution scaled by sqrt(2 / (input_dim + output_dim))

    Attributes
    ----------
    input_dim : int
       Size of the vocabulary.
    output_dim : int
       Embedding dimension.
    weights : Parameter
       Learnable embedding matrix of shape (input_dim, output_dim).
    inputs : Tensor
       Input tensor (one-hot encoded) from the last forward pass.
    outputs : Tensor
       Output tensor (dense embeddings) from the last forward pass.
    dX : Tensor
       Gradient of the loss with respect to the inputs.
    dW : ndarray
       Gradient of the loss with respect to the weights.

    Notes
    -----
    - Inputs are expected to be one-hot encoded along the last dimension.
    - For batched inputs or higher-dimensional inputs (e.g., sequences of one-hot vectors),
     the layer automatically flattens leading dimensions during computation and restores
     them in the output.
    - The embedding operation is implemented via matrix multiplication with one-hot vectors,
     which is mathematically equivalent to an embedding lookup but less efficient. In practice,
     frameworks often optimize this by using direct indexing.
    """
    def __init__(self, input_dim, output_dim, weight_initializer="he_uniform"):
        super().__init__()

        def initialize_weights(weight_init, input_dim, output_dim):
            """
            Initializes the weights based on the specified method.

            Parameters
            ----------
            weight_init : str
                Weight initialization method.
            input_dim : int
                Size of the vocabulary.
            output_dim : int
                Dimension of the embedding.

            Returns
            -------
            Tensor
                Initialized weights.
            """
            if weight_init.lower() == "he_normal":
                weights = Parameter(np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim), requires_grad=True)
            elif weight_init.lower() == "he_uniform":
                limit = np.sqrt(6 / input_dim)
                weights = Parameter(np.random.uniform(-limit, limit, (input_dim, output_dim)), requires_grad=True)
            elif weight_init.lower() == "glorot_normal":
                weights = Parameter(np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / (input_dim + output_dim)), requires_grad=True)
            elif weight_init.lower() == "glorot_uniform":
                limit = np.sqrt(6 / (input_dim + output_dim))
                weights = Parameter(np.random.uniform(-limit, limit, (input_dim, output_dim)), requires_grad=True)
            else:
                raise ValueError(
                    "`weight_init` must be either 'he_uniform', 'he_normal', 'glorot_uniform' or 'glorot_normal'.")
            return weights

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.leading_dims = ()

        self.inputs = None
        self.outputs = None
        self.parameters = [initialize_weights(weight_initializer, input_dim, output_dim)]
        self.dX = None
        self.dW = None

        self.flattened = False

    @property
    def weights(self):
        return self.parameters[0]

    @weights.setter
    def weights(self, value):
        self.parameters[0] = value

    def forward(self):
        """
        Convert one-hot encoded inputs to dense embeddings via matrix multiplication.

        Returns
        -------
        Tensor
           Embedding vectors with shape `(*leading_dims, output_dim)`
        """
        check_tensor(self.inputs, "inputs")
        check_tensor(self.outputs, "outputs", True)
        check_ohe(self.inputs, "inputs")

        if self.inputs.ndim > 2:
            self.leading_dims = self.inputs.shape[:-1]
            self.flattened = True
            self.inputs = flatten_tensor(self.inputs)

        self.inputs.requires_grad = True
        self.weights.requires_grad = True

        self.outputs = dot(self.inputs, self.weights)

        if self.flattened:
            input_shape = (*self.leading_dims, self.input_dim)
            output_shape = (*self.leading_dims, self.output_dim)
            self.inputs = restore_flattened(self.inputs, input_shape)
            self.outputs = restore_flattened(self.outputs, output_shape)

        return self.outputs

    def backward(self, dX):
        """
        Compute gradients through the embedding layer.

        Parameters
        ----------
        dX : Tensor
            Gradient of loss with respect to layer outputs (shape matches `outputs`)

        Returns
        -------
        Tensor
            Gradient of loss with respect to layer inputs (shape matches `inputs`)
        """
        self.dY = dX

        if self.flattened:
            dX = flatten_tensor(dX)
            self.inputs = flatten_tensor(self.inputs)
            self.outputs = flatten_tensor(self.outputs)

        self.outputs.backward(dX)
        self.dX = self.inputs.grad
        self.dW = self.weights.grad

        if self.flattened:
            dX_shape = (*self.leading_dims, self.input_dim)
            input_shape = (*self.leading_dims, self.input_dim)
            output_shape = (*self.leading_dims, self.output_dim)
            self.dX = restore_flattened(self.dX, dX_shape)
            self.inputs = restore_flattened(self.inputs, input_shape)
            self.outputs = restore_flattened(self.outputs, output_shape)

        return self.dX

    def zero_grad(self):
        self.outputs.zero_grad()

