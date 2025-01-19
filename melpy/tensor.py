import numpy as np

class Tensor:
    def __init__(self, object, requires_grad=False, *args, **kwargs):
        self.array = np.array(object, *args, **kwargs).astype(np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.array)

    @property
    def shape(self):
        return self.array.shape

    @property
    def size(self):
        return self.array.size

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def T(self):
        return Tensor(self.array.T)

    def __str__(self):
        return str(self.array)

    def __repr__(self):
        return f"Tensor({self.array})"

    def __add__(self, value):
        return add(self, value)

    def __radd__(self, value):
        return self.__add__(value)

    def __iadd__(self, value):
        self.array += value.array if isinstance(value, Tensor) else value
        return self

    def __sub__(self, value):
        return subtract(self, value)

    def __rsub__(self, value):
        return subtract(value, self)

    def __isub__(self, value):
        self.array -= value.array if isinstance(value, Tensor) else value
        return self

    def __neg__(self):
        return subtract(0, self)

    def __mul__(self, value):
        return multiply(self, value)

    def __rmul__(self, value):
        return self.__mul__(value)

    def __imul__(self, value):
        self.array *= value.array if isinstance(value, Tensor) else value
        return self

    def __truediv__(self, value):
        return divide(self.array, value)

    def __rtruediv__(self, value):
        return divide(value, self)

    def __itruediv__(self, value):
        self.array /= value.array if isinstance(value, Tensor) else value
        return self

    def __floordiv__(self, value):
        return floor_divide(self, value)

    def __matmul__(self, value):
        return matmul(self, value)

    def __rmatmul__(self, value):
        return self.__matmul__(value)

    def __imatmul__(self, value):
        self.array @= value.array if isinstance(value, Tensor) else value
        return self

    def __pow__(self, value):
        return power(self, value)

    def __ipow__(self, value):
        self.array **= value.array if isinstance(value, Tensor) else value
        return self

    def __eq__(self, value):
        return Tensor(self.array == (value.array if isinstance(value, Tensor) else value))

    def __ne__(self, value):
        return Tensor(self.array != (value.array if isinstance(value, Tensor) else value))

    def __gt__(self, value):
        return Tensor(self.array > (value.array if isinstance(value, Tensor) else value))

    def __ge__(self, value):
        return Tensor(self.array >= (value.array if isinstance(value, Tensor) else value))

    def __lt__(self, value):
        return Tensor(self.array < (value.array if isinstance(value, Tensor) else value))

    def __le__(self, value):
        return Tensor(self.array <= (value.array if isinstance(value, Tensor) else value))

    def __len__(self):
        return len(self.array)

    def reshape(self, *args, **kwargs):
        self.array = self.array.reshape(*args, **kwargs)
        return self

    def zero_grad(self):
        self.grad = np.zeros_like(self.array)

    def to_numpy(self):
        return self.array

class zeros(Tensor):
    def __init__(self, *args, **kwargs):
        array = np.zeros(*args, **kwargs)
        super().__init__(object = array)

class zeros_like(Tensor):
    def __init__(self, a,  *args, **kwargs):
        def _get_array(obj):
            if isinstance(obj, Operation):
                return obj.output.array
            elif isinstance(obj, Tensor):
                return obj.array
            else:
                return np.array(obj)

        array = np.zeros_like(_get_array(a), *args, **kwargs)
        super().__init__(object = array)

class Parameter(Tensor):
    def __init__(self, object, *args, **kwargs):
        super().__init__(object, *args, **kwargs)
        self.momentums = zeros_like(self)
        self.cache = zeros_like(self)

    def zero_grad(self):
        self.grad = np.zeros_like(self.array)
        self.momentums.zero_grad()
        self.cache.zero_grad()

class Operation:
    def __init__(self, x1, x2=None, *args, **kwargs):
        self.x1 = x1
        self.x2 = x2
        self.output = None
        self.__dict__.update(kwargs)
        self.forward(*args, **kwargs)

    @property
    def grad(self):
        if self.output is not None:
            return self.output.grad

    @grad.setter
    def grad(self, value):
        self.output.grad = value

    @property
    def array(self):
        if self.output is not None:
            return self.output.array

    @property
    def shape(self):
        if self.output is not None:
            return self.output.shape

    @property
    def size(self):
        if self.output is not None:
            return self.output.size

    @property
    def T(self):
        if self.output is not None:
            return self.output.T

    @property
    def ndim(self):
        if self.output is not None:
            return self.output.ndim

    @property
    def requires_grad(self):
        if self.output is not None:
            return self.output.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self.output.requires_grad = value

    def __str__(self):
        return self.output.array.__str__()

    def __repr__(self):
        return f"Tensor({self.array})"

    def __neg__(self):
        return subtract(0, self)

    def __add__(self, value):
        return add(self, value)

    def __radd__(self, value):
        return self.__add__(value)

    def __sub__(self, value):
        return subtract(self, value)

    def __rsub__(self, value):
        return subtract(value, self)

    def __mul__(self, value):
        return multiply(self, value)

    def __rmul__(self, value):
        return self.__mul__(value)

    def __truediv__(self, value):
        return divide(self, value)

    def __rtruediv__(self, value):
        return divide(value, self)

    def __floordiv__(self, value):
        return floor_divide(self, value)

    def __matmul__(self, value):
        return matmul(self, value)

    def __rmatmul__(self, value):
        return self.__matmul__(value)

    def __pow__(self, value):
        return power(self, value).output

    def forward(self, *args, **kwargs):
        pass

    def backward(self, grad):
        pass

    def zero_grad(self):
        if isinstance(self.x1, Operation) or isinstance(self.x1, Tensor) and self.x1.requires_grad:
            self.x1.zero_grad()
        if hasattr(self, "x2") and ((isinstance(self.x2, Operation) or isinstance(self.x2, Tensor))) and self.x2.requires_grad:
            self.x2.zero_grad()
        if self.output is not None:
            self.output.zero_grad()

    def _get_array(self, obj):
        if isinstance(obj, Operation):
            return obj.output.array
        elif isinstance(obj, Tensor):
            return obj.array
        else:
            return np.array(obj)

    def _apply_grad(self, obj, grad):
        if isinstance(obj, Function) and obj.requires_grad and not leaky_relu:
            obj.output.output.grad += grad
            obj.backward(grad)
        elif isinstance(obj, Operation) and obj.requires_grad:
            obj.output.grad += grad
            obj.backward(grad)
        elif isinstance(obj, Tensor) and obj.requires_grad:
            obj.grad += grad

    def _requires_grad(self, *inputs):
        return any((isinstance(i, Tensor) or isinstance(i, Operation)) and i.requires_grad for i in inputs)

    def _compress_grad(self, grad, tensor):
        grad = np.atleast_2d(grad)
        tensor = self._get_array(tensor)
        extra_dims = np.array(grad).ndim - tensor.ndim
        for _ in range(extra_dims):
            grad = np.sum(grad, axis=0)
        for i, dim in enumerate(tensor.shape):
            if dim == 1:
                grad = np.sum(grad, axis=i, keepdims=True)
        return grad

class sum(Operation):
    def __init__(self, x1, x2=None, *args, **kwargs):
        self.axis=None
        self.keepdims=False
        super().__init__(x1, x2, *args,  **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(np.sum(x1_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        if self.axis != None and self.keepdims == False:
            grad = np.broadcast_to(grad, self.x1.shape)
            self._apply_grad(self.x1, grad)
        else:
            self._apply_grad(self.x1, self._compress_grad(grad, self.x1))

class add(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        super().__init__(x1, x2, *args,  **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(np.add(x1_array, x2_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self._apply_grad(self.x1, self._compress_grad(grad, x1_array))
        self._apply_grad(self.x2, self._compress_grad(grad, x2_array))

class subtract(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        super().__init__(x1, x2, *args,  **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(np.subtract(x1_array, x2_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        self._apply_grad(self.x1, self._compress_grad(grad, self.x1))
        self._apply_grad(self.x2, -self._compress_grad(grad, self.x2))

class multiply(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        super().__init__(x1, x2, *args,  **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(np.multiply(x1_array, x2_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self._apply_grad(self.x1, grad * x2_array)
        self._apply_grad(self.x2, grad * x1_array)

class dot(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        super().__init__(x1, x2, *args, **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(np.dot(x1_array, x2_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self._apply_grad(self.x1, np.dot(grad, x2_array.T))
        self._apply_grad(self.x2, np.dot(x1_array.T, grad))

class matmul(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        super().__init__(x1, x2, *args,  **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(np.matmul(x1_array, x2_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self._apply_grad(self.x1, np.matmul(grad, x2_array.T))
        self._apply_grad(self.x2, np.matmul(x1_array.T, grad))

class divide(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        super().__init__(x1, x2, *args,  **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(np.divide(x1_array, x2_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self._apply_grad(self.x1, self._compress_grad(grad / x2_array, self.x1))
        self._apply_grad(self.x2, self._compress_grad(-grad * x1_array / (x2_array ** 2), self.x2))

class floor_divide(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        super().__init__(x1, x2, *args,  **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(np.floor_divide(x1_array, x2_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        self._apply_grad(self.x1, grad)
        self._apply_grad(self.x2, grad)

class power(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        super().__init__(x1, x2, *args,  **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(np.power(x1_array, x2_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self._apply_grad(self.x1, grad * x2_array * np.power(x1_array, x2_array - 1))
        self._apply_grad(self.x2, grad * np.power(x1_array, x2_array) * np.log(x1_array + 1e-15))

class exp(Operation):
    def __init__(self, x1, x2=None, *args, **kwargs):
        super().__init__(x1, x2, *args,  **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(np.exp(x1_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        self._apply_grad(self.x1, grad * np.exp(x1_array))

class log(Operation):
    def __init__(self, x1, x2=None, *args, **kwargs):
        super().__init__(x1, x2, *args,  **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(np.log(x1_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        self._apply_grad(self.x1, grad / x1_array)

class max(Operation):
    def __init__(self, x1, x2=None, axis=None, keepdims=False, *args, **kwargs):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x1, x2, axis, keepdims, *args,  **kwargs)

    def forward(self, axis, keepdims, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(np.max(x1_array, axis=axis, keepdims=keepdims, *args, **kwargs), requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        mask = (x1_array.max(axis=self.axis, keepdims=True) == x1_array).astype(int)
        self._apply_grad(self.x1, grad * mask)

class min(Operation):
    def __init__(self, x1, x2=None, axis=None, keepdims=False, *args, **kwargs):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x1, x2, *args,  **kwargs)

    def forward(self, axis, keepdims, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(np.min(x1_array, axis=axis, keepdims=keepdims, *args, **kwargs), requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        mask = (x1_array.min(axis=self.axis, keepdims=True) == x1_array).astype(int)
        self._apply_grad(self.x1, grad * mask)

class maximum(Operation):
    def __init__(self, x1, x2=None, *args, **kwargs):
        super().__init__(x1, x2,  *args, **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(np.maximum(x1_array, x2_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        output_array = self._get_array(self.output)
        mask1 = (output_array == x1_array).astype(int)
        mask2 = (output_array == x2_array).astype(int)
        self._apply_grad(self.x1, grad * mask1)
        self._apply_grad(self.x2, grad * mask2)

class minimum(Operation):
    def __init__(self, x1, x2=None, *args, **kwargs):
        super().__init__(x1, x2, *args, **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(np.minimum(x1_array, x2_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        output_array = self._get_array(self.output)
        mask1 = (output_array == x1_array).astype(int)
        mask2 = (output_array == x2_array).astype(int)
        self._apply_grad(self.x1, grad * mask1)
        self._apply_grad(self.x2, grad * mask2)

class argmax(Operation):
    def __init__(self, x1, x2=None, *args, **kwargs):
        super().__init__(x1, x2, *args,  **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(np.argmax(x1_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        self._apply_grad(self.x1, grad)

class argmin(Operation):
    def __init__(self, x1, x2=None, *args, **kwargs):
        super().__init__(x1, x2, *args,  **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(np.argmin(x1_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        self._apply_grad(self.x1, grad)

class clip(Operation):
    def __init__(self, x1, x2=None, *args, **kwargs):
        super().__init__(x1, x2, *args,  **kwargs)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(np.clip(x1_array, *args, **kwargs), requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        self._apply_grad(self.x1, grad)

class Function(Operation):
    def __init__(self, x1):
        super().__init__(x1)

    def update_requires_grad(self, obj, value):
        if not isinstance(value, bool):
            raise TypeError('`value` must be bool.')
        if isinstance(obj, Tensor) or isinstance(obj, Operation):
            obj.requires_grad = value
            self.forward()
        else:
            raise TypeError('`obj` must be Tensor or Operation.')

    def forward(self):
        pass

    def backward(self, grad):
        self.output.backward(grad)

    def derivative(self): # Must be used independently of a computational graph.
        x1 = self.x1
        self.x1 = Tensor(self.x1.array)
        self.update_requires_grad(self.x1, True)
        self.backward(1)
        grad = self.x1.grad
        self.x1 = x1
        return grad

class sigmoid(Function):
    def __init__(self, x1):
        super().__init__(x1)

    def forward(self):
        self.output = 1 / (1 + exp(-self.x1))
        return self.output.output

class tanh(Function):
    def __init__(self, x1):
        super().__init__(x1)

    def forward(self):
        self.output = (exp(self.x1) - exp(-self.x1)) / (exp(self.x1) + exp(-self.x1))
        return self.output.output

class softmax(Function):
    def __init__(self, x1):
        super().__init__(x1)

    def forward(self):
        exp_values = exp(self.x1 - max(self.x1, axis=1, keepdims=True))
        probabilities = exp_values / sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output.output

class relu(Function):
    def __init__(self, x1):
        super().__init__(x1)

    def forward(self):
        self.output = maximum(0, self.x1)
        return self.output.output

class leaky_relu(Function):
    def __init__(self, x1):
        super().__init__(x1)

    def forward(self):
        self.output = Tensor(np.where(self.x1.array > 0, self.x1.array, self.x1.array * 0.01), requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        self._apply_grad(self.x1, grad * self.derivative())

    def derivative(self):
        self.update_requires_grad(self.x1, True)
        dA = np.ones_like(self.x1.array)
        dA[self.x1.array < 0] = 0.01
        self.update_requires_grad(self.x1, False)
        return dA