import numpy as np

class Tensor:
    def __init__(self, object, requires_grad=True, *args, **kwargs):
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
    def T(self):
        return Tensor(self.array.T)

    def __str__(self):
        return str(self.array)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.array})"

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

    def zero_grad(self):
        self.grad = np.zeros_like(self.array)

    def to_numpy(self):
        return self.array

class Operation:
    def __init__(self,x1, *args, **kwargs):
        self.x1 = x1
        self.result = None
        self.forward()

    @property
    def grad(self):
        if self.result is not None:
            return self.result.grad

    def __str__(self):
        return self.result.array.__str__()

    def __repr__(self):
        return self.__str__()

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

    def __matmul__(self, value):
        return matmul(self, value)

    def __rmatmul__(self, value):
        return self.__matmul__(value)

    def __pow__(self, value):
        return power(self, value).result

    def forward(self):
        pass

    def backward(self, grad):
        pass

    def zero_grad(self):
        if isinstance(self.x1, Operation):
            self.x1.zero_grad()
        if hasattr(self, "x2") and isinstance(self.x2, Operation):
            self.x2.zero_grad()
        if self.result is not None:
            self.result.zero_grad()

    def _get_array(self, obj):
        if isinstance(obj, Operation):
            return obj.result.array
        elif isinstance(obj, Tensor):
            return obj.array
        else:
            return obj

    def _apply_grad(self, obj, grad):
        if isinstance(obj, Operation):
            obj.result.grad += grad
            obj.backward(grad)
        elif isinstance(obj, Tensor) and obj.requires_grad:
            obj.grad += grad

    def _requires_grad(self, *inputs):
        return any(isinstance(i, Tensor) and i.requires_grad for i in inputs)

class sum(Operation):
    def forward(self):
        x1_array = self._get_array(self.x1)
        self.result = Tensor(np.sum(x1_array), requires_grad=self._requires_grad(self.x1))
        return self.result

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        grad = grad * np.ones_like(x1_array)
        self._apply_grad(self.x1, grad)

class add(Operation):
    def __init__(self, x1, x2):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.result = Tensor(np.add(x1_array, x2_array), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.result

    def backward(self, grad):
        self._apply_grad(self.x1, grad)
        self._apply_grad(self.x2, grad)

class subtract(Operation):
    def __init__(self, x1, x2):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.result = Tensor(np.subtract(x1_array, x2_array), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.result

    def backward(self, grad):
        self._apply_grad(self.x1, grad)
        self._apply_grad(self.x2, -grad)


class multiply(Operation):
    def __init__(self, x1, x2):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.result = Tensor(np.multiply(x1_array, x2_array),requires_grad=self._requires_grad(self.x1, self.x2))
        return self.result

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self._apply_grad(self.x1, grad * x2_array)
        self._apply_grad(self.x2, grad * x1_array)

class matmul(Operation):
    def __init__(self, x1, x2):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.result = Tensor(np.matmul(x1_array, x2_array), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.result

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self._apply_grad(self.x1, grad * x2_array.T)
        self._apply_grad(self.x2, grad * x1_array.T)

class divide(Operation):
    def __init__(self, x1, x2):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.result = Tensor(np.divide(x1_array, x2_array), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.result

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self._apply_grad(self.x1, grad / x2_array)
        self._apply_grad(self.x2, -grad * x1_array / (x2_array ** 2))


class power(Operation):
    def __init__(self, x1, x2):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.result = Tensor(np.power(x1_array, x2_array), requires_grad=self._requires_grad(self.x1, self.x2))
        return self.result

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self._apply_grad(self.x1, grad * x2_array * np.power(x1_array, x2_array - 1))
        self._apply_grad(self.x2, grad * np.power(x1_array, x2_array) * np.log(x1_array))

class exp(Operation):
    def forward(self):
        x1_array = self._get_array(self.x1)
        self.result = Tensor(np.exp(x1_array), requires_grad=self._requires_grad(self.x1))
        return self.result

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        self._apply_grad(self.x1, grad * np.exp(x1_array))

class log(Operation):
    def forward(self):
        x1_array = self._get_array(self.x1)
        self.result = Tensor(np.log(x1_array), requires_grad=self._requires_grad(self.x1))
        return self.result

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        self._apply_grad(self.x1, grad / x1_array)

