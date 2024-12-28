import numpy as np

class Tensor:
    def __init__(self, object, requires_grad=True, *args, **kwargs):
        self.array = np.array(object, *args, **kwargs).astype(np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.array)

    def __str__(self):
        return str(self.array)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.array})"

    def __add__(self, value):
        return Tensor(self.array + value.array if isinstance(value, Tensor) else self.array + value)

    def __radd__(self, value):
        return self.__add__(value)

    def __iadd__(self, value):
        self.array += value.array if isinstance(value, Tensor) else value
        return self

    def __sub__(self, value):
        return Tensor(self.array - value.array if isinstance(value, Tensor) else self.array - value)

    def __rsub__(self, value):
        return Tensor(value - self.array if isinstance(value, Tensor) else value - self.array)

    def __isub__(self, value):
        self.array -= value.array if isinstance(value, Tensor) else value
        return self

    def __neg__(self):
        return Tensor(-self.array)

    def __mul__(self, value):
        return Tensor(self.array * value.array if isinstance(value, Tensor) else self.array * value)

    def __rmul__(self, value):
        return self.__mul__(value)

    def __imul__(self, value):
        self.array *= value.array if isinstance(value, Tensor) else value
        return self

    def __truediv__(self, value):
        return Tensor(self.array / value.array if isinstance(value, Tensor) else self.array / value)

    def __rtruediv__(self, value):
        return Tensor(value / self.array if isinstance(value, Tensor) else value / self.array)

    def __itruediv__(self, value):
        self.array /= value.array if isinstance(value, Tensor) else value
        return self

    def __matmul__(self, value):
        return Tensor(self.array @ value.array if isinstance(value, Tensor) else self.array @ value)

    def __rmatmul__(self, value):
        return self.__matmul__(value)

    def __imatmul__(self, value):
        self.array @= value.array if isinstance(value, Tensor) else value
        return self

    def __pow__(self, value):
        return Tensor(self.array ** value.array if isinstance(value, Tensor) else self.array ** value)

    def __ipow__(self, value):
        self.array **= value.array if isinstance(value, Tensor) else value
        return self

    def __len__(self):
        return len(self.array)

    def zero_grad(self):
        self.grad = np.zeros_like(self.array)

    def T(self):
        return Tensor(self.array.T)

class Operation:
    def __init__(self,x1, *args, **kwargs):
        self.x1 = x1
        self.result = None
        self.forward(*args, **kwargs)

    @property
    def grad(self):
        if self.result is not None:
            return self.result.grad

    def __str__(self):
        return self.result.array.__str__()

    def __repr__(self):
        if hasattr(self, "x2"):
            return f"{self.__class__.__name__}({self.x1.__repr__()}, {self.x2.__repr__()})"
        return f"{self.__class__.__name__}({self.x1.__repr__()})"

    def __neg__(self):
        return -self.result.array

    def forward(self, *args, **kwargs):
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

class sum(Operation):
    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        self.result = Tensor(np.sum(x1_array), requires_grad=(isinstance(self.x1, Tensor) \
                                                              and self.x1.requires_grad))
        return self.result

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        grad = grad * np.ones_like(x1_array)
        self._apply_grad(self.x1, grad)

class add(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.result = Tensor(np.add(x1_array, x2_array), requires_grad=((isinstance(self.x1, Tensor) \
                                                                         and self.x1.requires_grad) \
                                                                         or (isinstance(self.x2, Tensor) \
                                                                         and self.x2.requires_grad)))
        return self.result

    def backward(self, grad):
        self._apply_grad(self.x1, grad)
        self._apply_grad(self.x2, grad)

class subtract(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.result = Tensor(np.subtract(x1_array, x2_array), requires_grad=((isinstance(self.x1, Tensor) \
                                                                              and self.x1.requires_grad) \
                                                                              or (isinstance(self.x2, Tensor) \
                                                                              and self.x2.requires_grad)))
        return self.result

    def backward(self, grad):
        self._apply_grad(self.x1, grad)
        self._apply_grad(self.x2, -grad)


class multiply(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.result = Tensor(np.multiply(x1_array, x2_array), requires_grad=((isinstance(self.x1, Tensor) \
                                                                            and self.x1.requires_grad) \
                                                                            or (isinstance(self.x2, Tensor) \
                                                                            and self.x2.requires_grad)))
        return self.result

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self._apply_grad(self.x1, grad * x2_array)
        self._apply_grad(self.x2, grad * x1_array)

class divide(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.result = Tensor(np.divide(x1_array, x2_array), requires_grad=((isinstance(self.x1, Tensor) \
                                                                            and self.x1.requires_grad) \
                                                                            or (isinstance(self.x2, Tensor) \
                                                                            and self.x2.requires_grad)))
        return self.result

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self._apply_grad(self.x1, grad / x2_array)
        self._apply_grad(self.x2, -grad * x1_array / (x2_array ** 2))


class power(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.result = Tensor(np.power(x1_array, x2_array), requires_grad=((isinstance(self.x1, Tensor) \
                                                                           and self.x1.requires_grad) \
                                                                           or (isinstance(self.x2, Tensor) \
                                                                           and self.x2.requires_grad)))
        return self.result

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self._apply_grad(self.x1, grad * x2_array * np.power(x1_array, x2_array - 1))
        self._apply_grad(self.x2, grad * np.power(x1_array, x2_array) * np.log(x1_array))

class exp(Operation):
    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        self.result = Tensor(np.exp(x1_array), requires_grad=(isinstance(self.x1, Tensor) \
                                                              and self.x1.requires_grad))
        return self.result

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        self._apply_grad(self.x1, grad * np.exp(x1_array))

class log(Operation):
    def forward(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        self.result = Tensor(np.log(x1_array), requires_grad=(isinstance(self.x1, Tensor) \
                                                              and self.x1.requires_grad))
        return self.result

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        self._apply_grad(self.x1, grad / x1_array)


