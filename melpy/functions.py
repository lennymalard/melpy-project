import numpy as np
from math import sqrt
from .im2col import *

class Layer:
    """
    Base class for all layers in the neural network.

    Attributes
    ----------
    inputs : ndarray
        Input data.
    targets : ndarray
        Target data.
    outputs : ndarray
        Output data.
    weights : ndarray
        Weights of the layer.
    biases : ndarray
        Biases of the layer.
    dX : ndarray
        Partial derivative of input with respect to loss.
    dY : ndarray
        Partial derivative of output with respect to loss.
    dW : ndarray
        Partial derivative of weight with respect to loss.
    dB : ndarray
        Partial derivative of bias with respect to loss.

    Methods
    -------
    derivative()
        Computes the derivative of the layer.
    forward()
        Computes the forward pass of the layer.
    backward()
        Computes the backward pass of the layer.
    """
    def __init__(self):
        self.inputs = None
        self.targets = None
        self.outputs = None
        self.weights = None
        self.biases = None
        self.dX = None
        self.dY = None
        self.dW = None
        self.dB = None

    def derivative(self):
        """
        Computes the derivative of the layer.
        """
        pass

    def forward(self):
        """
        Computes the forward pass of the layer.
        """
        pass

    def backward(self, dX):
        """
        Computes the backward pass of the layer.
        """
        pass

class Loss:
    """
    Base class for all loss functions.

    Methods
    -------
    loss(targets : ndarray, outputs : ndarray)
        Computes the loss.
    derivative()
        Computes the derivative of the loss.
    """
    def __init__(self):
        pass

    def loss(self, targets, outputs):
        """
        Computes the loss.
        """
        pass

    def derivative(self, targets, outputs):
        """
        Computes the derivative of the loss.
        """
        pass

class MSE(Loss):
    """
    A class to perform the mean squared error loss.

    Methods
    -------
    loss(targets : ndarray, outputs : ndarray)
        Computes the mean squared error.
    """
    def __init__(self):
        super().__init__()

    def loss(self, targets, outputs):
        """
        Computes the mean squared error.

        Parameters
        ----------
        targets : ndarray
            Target data.
        outputs : ndarray
            Output data.

        Returns
        -------
        float
            The mean squared error.
        """
        diff = targets - outputs
        return np.sum(diff * diff) / np.size(diff)

class BinaryCrossEntropy(Loss):
    """
    A class to perform binary cross entropy loss.

    Methods
    -------
    loss(targets : ndarray, outputs : ndarray)
        Computes the binary cross entropy loss.
    derivative(targets, outputs)
        Computes the binary cross entropy derivative.
    """
    def __init__(self):
        super().__init__()

    def loss(self, targets, outputs):
        """
        Computes the binary cross entropy loss.

        Parameters
        ----------
        targets : ndarray
            Target data.
        outputs : ndarray
            Output data.

        Returns
        -------
        float
            The binary cross entropy loss.
        """
        e = 1e-10
        return -(np.sum(targets * np.log(outputs + e) + \
                        (1-targets) * np.log(1-outputs + e))) / len(targets)

    def derivative(self, targets, outputs):
        """
        Computes the binary cross entropy derivative.

        Parameters
        ----------
        targets : ndarray
            Target data.
        outputs : ndarray
            Output data.

        Returns
        -------
        ndarray
            The binary cross entropy derivative.
        """
        e = 1e-10
        return -(targets / outputs - (1 - targets + e) / (1 - outputs + e)) / len(outputs)

class CategoricalCrossEntropy(Loss):
    """
    A class to perform categorical cross entropy loss.

    Methods
    -------
    loss(targets : ndarray, outputs : ndarray)
        Computes the categorical cross entropy loss.
    derivative(targets : ndarray, outputs : ndarray)
        Computes the categorical cross entropy derivative.
    """
    def __init__(self):
        super().__init__()

    def loss(self, targets, outputs):
        """
        Computes the categorical cross entropy loss.

        Parameters
        ----------
        targets : ndarray
            Target data.
        outputs : ndarray
            Output data.

        Returns
        -------
        float
            The categorical cross entropy loss.
        """
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
        """
        Computes the categorical cross entropy derivative.

        Parameters
        ----------
        targets : ndarray
            Target data.
        outputs : ndarray
            Output data.

        Returns
        -------
        ndarray
            The categorical cross entropy derivative.
        """
        if len(targets.shape) == 1:
            targets = np.eye(len(outputs[0]))[targets]
        return ((-targets + 1e-5) / (outputs + 1e-5)) / len(outputs)

class Dense(Layer):
    """
    A class that performs dense layer operations.

    Attributes
    ----------
    inputs : ndarray
        Input data.
    outputs : ndarray
        Output data.
    dX : ndarray
        Partial derivative of input with respect to loss.
    dY : ndarray
        Partial derivative of output with respect to loss.
    dW : ndarray
        Partial derivative of weight with respect to loss.
    dB : ndarray
        Partial derivative of bias with respect to loss.

    Methods
    -------
    forward()
        Computes the dense layer forward pass.
    backward(dX : ndarray)
        Computes the dense layer backward pass.
    """
    def __init__(self, n_in, n_out, weight_initializer="he_normal"):
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

        self.weights = initialize_weights(weight_initializer, n_in, n_out)
        self.biases = np.random.rand(1, n_out)
        self.w_momentum = np.zeros_like(self.weights)
        self.b_momentum = np.zeros_like(self.biases)

    def forward(self):
        """
        Computes the dense layer forward pass.

        Returns
        -------
        ndarray
            The output data.
        """
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, dX):
        """
        Computes the dense layer backward pass.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of output data (dY -> dX) with respect to loss.

        Returns
        -------
        ndarray
            Partial derivative of input data with respect to loss.
        """
        self.dY = dX
        self.dW = np.dot(self.inputs.T, self.dY)
        self.dB = np.sum(self.dY, axis=0, keepdims=True)
        self.dX = np.dot(self.dY, self.weights.T)

        return self.dX

class ReLU(Layer):
    """
    A class that performs ReLU layer operations.

    Attributes
    ----------
    inputs : ndarray
        Input data.
    outputs : ndarray
        Output data.
    dX : ndarray
        Partial derivative of input with respect to loss.
    dY : ndarray
        Partial derivative of output with respect to loss.

    Methods
    -------
    derivative()
        Computes the ReLU derivative.
    forward()
        Computes the ReLU forward pass.
    backward(dX)
        Computes the ReLU backward pass.
    """
    def __init__(self):
        super().__init__()

    def derivative(self):
        """
        Computes the ReLU derivative.

        Returns
        -------
        ndarray
            The differentiated input data.
        """
        dA = np.ones_like(self.inputs)
        dA[self.inputs <= 0] = 0
        return dA

    def forward(self):
        """
        Computes the ReLU forward pass.

        Returns
        -------
        ndarray
            The output data.
        """
        self.outputs = np.maximum(0, self.inputs)
        return self.outputs

    def backward(self, dX):
        """
        Computes the ReLU backward pass.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of output data (dY -> dX) with respect to loss.

        Returns
        -------
        ndarray
            Partial derivative of input data with respect to loss.
        """
        self.dY = dX
        self.dX = self.dY * self.derivative()
        return self.dX

class LeakyReLU(Layer):
    """
    A class that performs Leaky ReLU layer operations.

    Attributes
    ----------
    inputs : ndarray
        Input data.
    outputs : ndarray
        Output data.
    dX : ndarray
        Partial derivative of input with respect to loss.
    dY : ndarray
        Partial derivative of output with respect to loss.

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
        super().__init__()

    def derivative(self):
        """
        Computes the Leaky ReLU derivative.

        Returns
        -------
        ndarray
            The differentiated input data with respect to loss.
        """
        dA = np.ones_like(self.inputs)
        dA[self.inputs <= 0] = 0.01
        return dA

    def forward(self):
        """
        Computes the Leaky ReLU forward pass.

        Returns
        -------
        ndarray
            The output data.
        """
        self.outputs = np.where(self.inputs > 0, self.inputs, self.inputs * 0.01)
        return self.outputs

    def backward(self, dX):
        """
        Computes the Leaky ReLU backward pass.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of output data (dY -> dX) data with respect to loss.

        Returns
        -------
        ndarray
            Partial derivative of input data with respect to loss.
        """
        self.dY = dX
        self.dX = self.dY * self.derivative()
        return self.dX

class Sigmoid(Layer):
    """
    A class that performs Sigmoid layer operations.

    Attributes
    ----------
    inputs : ndarray
        Input data.
    outputs : ndarray
        Output data.
    dX : ndarray
        Partial derivative of input with respect to loss.
    dY : ndarray
        Partial derivative of output with respect to loss.

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
        ndarray
            The output data.
        """
        self.outputs = 1 / (1 + np.exp(-self.inputs))
        return self.outputs

    def backward(self, dX):
        """
        Computes the Sigmoid backward pass.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of output data (dY -> dX) data with respect to loss.

        Returns
        -------
        ndarray
            Partial derivative of input data with respect to loss.
        """
        self.dY = dX
        self.dX = self.dY * self.derivative()
        return self.dX

class Softmax(Layer):
    """
    A class that performs Softmax layer operations.

    Attributes
    ----------
    inputs : ndarray
        Input data.
    outputs : ndarray
        Output data.
    dX : ndarray
        Partial derivative of input with respect to loss.
    dY : ndarray
        Partial derivative of output with respect to loss.

    Methods
    -------
    forward()
        Computes the Softmax forward pass.
    backward(dX : ndarray)
        Computes the Softmax backward pass.
    """
    def __init__(self):
        super().__init__()

    def forward(self):
        """
        Computes the Softmax forward pass.

        Returns
        -------
        ndarray
            The output data.
        """
        exp_values = np.exp(self.inputs - np.max(self.inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities
        return self.outputs

    def backward(self, dX):
        """
        Computes the Softmax backward pass.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of output data (dY -> dX) data with respect to loss.

        Returns
        -------
        ndarray
            Partial derivative of input data with respect to loss.
        """
        self.dY = dX
        self.dX = np.empty_like(dX)
        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, self.dX)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dX[index] = np.dot(jacobian_matrix, single_dvalues)
        return self.dX

class Convolution2D(Layer):
    """
    A class that performs 2D convolution layer operations.

    Attributes
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    weights : ndarray
        Weights of the convolution layer.
    padding : str
        Padding type ('valid' or 'same').
    kernel_size : int
        Size of the convolution kernel.
    stride : int
        Stride of the convolution.
    w_momentum : ndarray
        Momentum for weights.

    Methods
    -------
    calculate_padding()
        Calculates the padding for the input.
    explicit_padding()
        Applies explicit padding to the input.
    get_output_size(input_height, input_width)
        Gets the output size of the convolution.
    forward()
        Computes the forward pass of the convolution layer.
    backward(dX : ndarray)
        Computes the backward pass of the convolution layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding="valid", stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = np.random.uniform(
            -sqrt(6 / (in_channels + out_channels)),
            sqrt(6 / (in_channels + out_channels)),
            (out_channels, in_channels, kernel_size, kernel_size)
        )
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.w_momentum = np.zeros_like(self.weights)

        if self.padding not in ["valid", "same"]:
            raise ValueError("invalid value for 'padding'")

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

            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left

            return (pad_top, pad_bottom, pad_left, pad_right)

    def explicit_padding(self):
        """
        Applies explicit padding to the input.

        Returns
        -------
        ndarray
            Padded input data.
        """
        pad_top, pad_bottom, pad_left, pad_right = self.calculate_padding()
        return np.pad(self.inputs, ((0, 0), (0, 0), (pad_top, pad_bottom),
                                    (pad_left, pad_right)), mode='constant')

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
        ndarray
            The output data.
        """
        self.input_padded = self.explicit_padding()

        self.input_cols = im2col(self.input_padded, self.kernel_size, self.stride)
        self.filter_cols = self.weights.reshape(self.out_channels, -1)

        output_height, output_width = self.get_output_size(self.inputs.shape[2], self.inputs.shape[3])

        self.output_cols = self.filter_cols @ self.input_cols

        self.outputs = np.array(np.hsplit(self.output_cols, self.inputs.shape[0])).reshape(
            (self.input_padded.shape[0], self.out_channels, output_height, output_width)
        )

        return self.outputs

    def backward(self, dX):
        """
        Computes the backward pass of the convolution layer.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of output data (dY -> dX) with respect to loss.

        Returns
        -------
        ndarray
            Partial derivative of input data with respect to loss.
        """
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
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode

        if self.mode not in ["max"]:
            raise ValueError("invalid value for 'mode'")

    def forward(self):
        """
        Computes the forward pass of the pooling layer.

        Returns
        -------
        ndarray
            The output data.
        """
        output_height = int((self.inputs.shape[2] - self.pool_size + self.stride) // self.stride)
        output_width = int((self.inputs.shape[3] - self.pool_size + self.stride) // self.stride)

        output_shape = (self.inputs.shape[0], self.inputs.shape[1], output_height, output_width)

        self.input_cols = im2col(self.inputs, self.pool_size, self.stride)
        self.input_cols_reshaped = np.array(np.hsplit(np.array(np.hsplit(self.input_cols, self.inputs.shape[0])), self.inputs.shape[1]))

        self.maxima = np.max(self.input_cols_reshaped, axis=2)
        self.maxima_reshaped = self.maxima.reshape(self.inputs.shape[1], -1)

        self.outputs = col2im(self.maxima_reshaped, output_shape, 1, 1)

        return self.outputs

    def backward(self, dX):
        """
        Computes the backward pass of the pooling layer.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of output data (dY -> dX) with respect to loss.

        Returns
        -------
        ndarray
            Partial derivative of input data with respect to loss.
        """
        self.dY = dX
        self.dX = np.zeros_like(self.inputs)

        self.dY_cols = im2col(self.dY, 1, 1)
        self.dY_cols_reshaped = np.array(np.hsplit(np.array(np.hsplit(self.dY_cols, self.dY.shape[0])), self.dY.shape[1])).transpose(0, 1, 3, 2)

        self.input_cols = im2col(self.inputs, self.pool_size, self.stride)
        self.input_cols_reshaped = np.array(np.hsplit(np.array(np.hsplit(self.input_cols, self.inputs.shape[0])), self.inputs.shape[1])).transpose(0, 1, 3, 2)

        self.output_cols = im2col(self.outputs, 1, 1)
        self.output_cols_reshaped = np.array(np.hsplit(np.array(np.hsplit(self.output_cols, self.inputs.shape[0])), self.inputs.shape[1])).transpose(0, 1, 3, 2)

        self.mask = np.array(self.input_cols_reshaped == self.output_cols_reshaped, dtype=np.uint64)

        self.dX_cols = np.concatenate(np.concatenate(np.array(self.mask * self.dY_cols_reshaped).transpose(0, 1, 3, 2), axis=1), axis=1)
        self.dX = col2im(self.dX_cols, self.inputs.shape, self.pool_size, self.stride)

        return self.dX

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
        super().__init__()

    def forward(self):
        """
        Computes the forward pass of the flattening layer.

        Returns
        -------
        ndarray
            The output data.
        """
        self.outputs = self.inputs.reshape((self.inputs.shape[0], -1))
        return self.outputs

    def backward(self, dX):
        """
        Computes the backward pass of the flattening layer.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of output data (dY -> dX) with respect to loss.

        Returns
        -------
        ndarray
            Partial derivative of input data with respect to loss.
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
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1")
        self.p = p
        self.mask = None
        self.training = True

    def forward(self):
        """
        Computes the forward pass of the dropout layer.

        Returns
        -------
        ndarray
            The output data.
        """
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, size=self.inputs.shape)
            self.outputs = self.inputs * self.mask * 1.0 / (1.0 - self.p)
        else:
            self.outputs = self.inputs
        return self.outputs

    def backward(self, dX):
        """
        Computes the backward pass of the dropout layer.

        Parameters
        ----------
        dX : ndarray
            Partial derivative of output data (dY -> dX) with respect to loss.

        Returns
        -------
        ndarray
            Partial derivative of input data with respect to loss.
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
        self.weights = np.random.rand(n_in, n_out)
        self.biases = np.random.rand(1, n_out)

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
