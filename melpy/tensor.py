import numpy as np

class Tensor:
    def __init__(self, object, requires_grad=True, *args, **kwargs):
        self.array = np.array(object, *args, **kwargs).astype(np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.array)

    def __str__(self):
        return self.array.__str__()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.array})"

    def __add__(self, value):
        if isinstance(value, Tensor):
            return Tensor(self.array + value.array)
        else:
            return Tensor(self.array + value)

    def __radd__(self, value):
        if isinstance(value, Tensor):
            return Tensor(self.array + value.array)
        else:
            return Tensor(self.array + value)

    def __iadd__(self, value):
        if isinstance(value, Tensor):
            self.array += value.array
        else:
            self.array += value
        return self

    def __sub__(self, value):
        if isinstance(value, Tensor):
            return Tensor(self.array - value.array)
        else:
            return Tensor(self.array - value)

    def __rsub__(self, value):
        if isinstance(value, Tensor):
            return Tensor(self.array - value.array)
        else:
            return Tensor(self.array - value)

    def __isub__(self, value):
        if isinstance(value, Tensor):
            self.array -= value.array
        else:
            self.array -= value
        return self

    def __neg__(self):
        return Tensor(self.array * -1)

    def __mul__(self, value):
        if isinstance(value, Tensor):
            return Tensor(self.array * value.array)
        else:
            return Tensor(self.array * value)

    def __rmul__(self, value):
        if isinstance(value, Tensor):
            return Tensor(self.array * value.array)
        else:
            return Tensor(self.array * value)

    def __imul__(self, value):
        if isinstance(value, Tensor):
            self.array *= value.array
        else:
            self.array *= value
        return self

    def __truediv__(self, value):
        if isinstance(value, Tensor):
            return Tensor(self.array / value.array)
        else:
            return Tensor(self.array / value)

    def __rtruediv__(self, value):
        if isinstance(value, Tensor):
            return Tensor(self.array / value.array)
        else:
            return Tensor(self.array / value)

    def __itruediv__(self, value):
        if isinstance(value, Tensor):
            self.array = self.array.astype(np.float64) / value.array.astype(np.float64)
        else:
            self.array = self.array.astype(np.float64) / value
        return self

    def __matmul__(self, value):
        if isinstance(value, Tensor):
            return Tensor(self.array @ value.array)
        else:
            return Tensor(self.array @ value)

    def __rmatmul__(self, value):
        if isinstance(value, Tensor):
            return Tensor(self.array @ value.array)
        else:
            return Tensor(self.array @ value)

    def __imatmul__(self, value):
        if isinstance(value, Tensor):
            self.array @= value.array
        else:
            self.array @= value
        return self

    def __pow__(self, value):
        if isinstance(value, Tensor):
            return Tensor(self.array ** value.array)
        else:
            return Tensor(self.array ** value)

    def __ipow__(self, value):
        if isinstance(value, Tensor):
            self.array **= value.array
        else:
            self.array **= value
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
        return self.result.__repr__()

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

class sum(Operation):
    def __init__(self, x1, *args, **kwargs):
        super().__init__(x1)

    def forward(self, *args, **kwargs):
        x1_array = self.x1.result.array if isinstance(self.x1, Operation) else self.x1.array if isinstance(self.x1, Tensor) else self.x1
        self.result = Tensor(np.sum(x1_array))
        return self.result

    def backward(self, grad):
        if isinstance(self.x1, Operation):
            self.x1.result.grad += grad
            self.x1.backward(self.x1.result.grad)
        elif isinstance(self.x1, Tensor) and self.x1.requires_grad:
            self.x1.grad += grad

class add(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self, *args, **kwargs):
        x1_array = self.x1.result.array if isinstance(self.x1, Operation) else self.x1.array if isinstance(self.x1, Tensor) else self.x1
        x2_array = self.x2.result.array if isinstance(self.x2, Operation) else self.x2.array if isinstance(self.x2, Tensor) else self.x2

        self.result = Tensor(np.add(x1_array, x2_array))
        return self.result

    def backward(self, grad):
        if isinstance(self.x1, Operation):
            self.x1.result.grad += grad
            self.x1.backward(self.x1.result.grad)
        elif isinstance(self.x1, Tensor) and self.x1.requires_grad:
            self.x1.grad += grad
        if isinstance(self.x2, Operation):
            self.x2.result.grad += grad
            self.x2.backward(self.x2.result.grad)
        elif isinstance(self.x2, Tensor) and self.x2.requires_grad:
            self.x2.grad += grad

class subtract(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self, *args, **kwargs):
        x1_array = self.x1.result.array if isinstance(self.x1, Operation) else self.x1.array if isinstance(self.x1, Tensor) else self.x1
        x2_array = self.x2.result.array if isinstance(self.x2, Operation) else self.x2.array if isinstance(self.x2, Tensor) else self.x2
        self.result = Tensor(np.subtract(x1_array - x2_array))
        return self.result

    def backward(self, grad):
        if isinstance(self.x1, Operation):
            self.x1.result.grad += grad
            self.x1.backward(self.x1.result.grad)
        elif isinstance(self.x1, Tensor) and self.x1.requires_grad:
            self.x1.grad += grad

        if isinstance(self.x2, Operation):
            self.x2.result.grad -= grad
            self.x2.backward(self.x2.result.grad)
        elif isinstance(self.x2, Tensor) and self.x2.requires_grad:
            self.x2.grad -= grad


class multiply(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self, *args, **kwargs):
        x1_array = self.x1.result.array if isinstance(self.x1, Operation) else self.x1.array if isinstance(self.x1, Tensor) else self.x1
        x2_array = self.x2.result.array if isinstance(self.x2, Operation) else self.x2.array if isinstance(self.x2, Tensor) else self.x2
        self.result = Tensor(np.multiply(x1_array, x2_array))
        return self.result

    def backward(self, grad):
        if isinstance(self.x1, Operation):
            self.x1.result.grad += grad * (self.x2.result.array if isinstance(self.x2, Operation) else self.x2.array if isinstance(self.x2, Tensor) else self.x2)
            self.x1.backward(self.x1.result.grad)
        elif isinstance(self.x1, Tensor) and self.x1.requires_grad:
            self.x1.grad += grad *(self.x2.result.array if isinstance(self.x2, Operation) else self.x2.array if isinstance(self.x2, Tensor) else self.x2)

        if isinstance(self.x2, Operation):
            self.x2.result.grad += grad * (self.x1.result.array if isinstance(self.x1, Operation) else self.x1.array if isinstance(self.x1, Tensor) else self.x1)
            self.x2.backward(self.x2.result.grad)
        elif isinstance(self.x2, Tensor) and self.x2.requires_grad:
            self.x2.grad += grad * (self.x1.result.array if isinstance(self.x1, Operation) else self.x1.array if isinstance(self.x1, Tensor) else self.x1)


class divide(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self, *args, **kwargs):
        x1_array = self.x1.result.array if isinstance(self.x1, Operation) else self.x1.array if isinstance(self.x1, Tensor) else self.x1
        x2_array = self.x2.result.array if isinstance(self.x2, Operation) else self.x2.array if isinstance(self.x2, Tensor) else self.x2
        self.result = Tensor(np.divide(x1_array, x2_array))
        return self.result

    def backward(self, grad):
        x2_squared = (self.x2.result.array if isinstance(self.x2, Operation) else self.x2.array if isinstance(self.x2, Tensor) else self.x2) ** 2
        if isinstance(self.x1, Operation):
            self.x1.result.grad += grad / (self.x2.result.array if isinstance(self.x2, Operation) else self.x2.array if isinstance(self.x2, Tensor) else self.x2)
            self.x1.backward(self.x1.result.grad)
        elif isinstance(self.x1, Tensor) and self.x1.requires_grad:
            self.x1.grad += grad / (self.x2.result.array if isinstance(self.x2, Operation) else self.x2.array if isinstance(self.x2, Tensor) else self.x2)

        if isinstance(self.x2, Operation):
            self.x2.result.grad += grad * -(self.x1.result.array if isinstance(self.x1, Operation) else self.x1.array if isinstance(self.x1, Tensor) else self.x1) / x2_squared
            self.x2.backward(self.x2.result.grad)
        elif isinstance(self.x2, Tensor) and self.x2.requires_grad:
            self.x2.grad += grad * -(self.x1.result.array if isinstance(self.x1, Operation) else self.x1.array if isinstance(self.x1, Tensor) else self.x1) / x2_squared


class power(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, x2)

    def forward(self, *args, **kwargs):
        x1_array = self.x1.result.array if isinstance(self.x1, Operation) else self.x1.array if isinstance(self.x1, Tensor) else self.x1
        x2_array = self.x2.result.array if isinstance(self.x2, Operation) else self.x2.array if isinstance(self.x2, Tensor) else self.x2
        self.result = Tensor(np.power(x1_array, x2_array))
        return self.result

    def backward(self, grad):
        x1_array = self.x1.result.array if isinstance(self.x1, Operation) else self.x1.array if isinstance(self.x1,Tensor) else self.x1
        x2_array = self.x2.result.array if isinstance(self.x2, Operation) else self.x2.array if isinstance(self.x2,Tensor) else self.x2

        if isinstance(self.x1, Operation):
            self.x1.result.grad += grad * x2_array * np.power(x1_array, x2_array - 1)
            self.x1.backward(self.x1.result.grad)
        elif isinstance(self.x1, Tensor) and self.x1.requires_grad:
            self.x1.grad += grad * x2_array * np.power(x1_array, x2_array - 1)

        if isinstance(self.x2, Operation):
            self.x2.result.grad += grad * np.power(x1_array, x2_array) * np.log(x1_array)
            self.x2.backward(self.x2.result.grad)
        elif isinstance(self.x2, Tensor) and self.x2.requires_grad:
            self.x2.grad += grad * np.power(x1_array, x2_array) * np.log(x1_array)
    
class exp(Operation):
    def __init__(self, x1, *args, **kwargs):
        super().__init__(x1)

    def forward(self, *args, **kwargs):
        if isinstance(self.x1, Operation):
            x1_array = self.x1.result.array
        elif isinstance(self.x1, Tensor):
            x1_array = self.x1.array
        else:
            x1_array = self.x1
        self.result = Tensor(np.exp(x1_array))
        return self.result

    def backward(self, grad):
        if isinstance(self.x1, Operation):
            x1_array = self.x1.result.array
            self.x1.result.grad += grad * np.exp(x1_array)
            self.x1.backward(self.x1.result.grad)
        elif isinstance(self.x1, Tensor) and self.x1.requires_grad:
            self.x1.grad += grad * np.exp(self.x1.array)

class log(Operation):
    def __init__(self, x1, *args, **kwargs):
        super().__init__(x1)

    def forward(self, *args, **kwargs):
        if isinstance(self.x1, Operation):
            x1_array = self.x1.result.array
        elif isinstance(self.x1, Tensor):
            x1_array = self.x1.array
        else:
            x1_array = self.x1

        self.result = Tensor(np.log(x1_array))
        return self.result

    def backward(self, grad):
        if isinstance(self.x1, Operation):
            x1_array = self.x1.result.array
            self.x1.result.grad += grad / x1_array
            self.x1.backward(self.x1.result.grad)
        elif isinstance(self.x1, Tensor) and self.x1.requires_grad:
            self.x1.grad += grad / self.x1.array


