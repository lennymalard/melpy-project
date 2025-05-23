import numpy as np
from copy import deepcopy

_F64_TINY = np.finfo(np.float64).tiny
_F64_MAX = np.finfo(np.float64).max
_F64_MAX_EXP_ARG = np.log(_F64_MAX).astype(np.float64)

class Tensor:
    def __init__(self, object, requires_grad=False, _operation=None, *args, **kwargs):
        self.array = np.array(object if not isinstance(object, Tensor) else object.array, *args, **kwargs).astype(np.float64)
        self._grad = None
        self._op = _operation
        self.requires_grad = requires_grad

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

    @property
    def grad(self):
        if self._grad is None:
            self._grad = Tensor(np.zeros_like(self.array))
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    def __str__(self):
        return str(self.array)

    def __repr__(self):
        return f"Tensor({self.array})"

    def __add__(self, value):
        return add(self, value)

    def __radd__(self, value):
        return self.__add__(value)

    def __iadd__(self, value):
        return self.__add__(value)

    def __sub__(self, value):
        return subtract(self, value)

    def __rsub__(self, value):
        return subtract(value, self)

    def __isub__(self, value):
        return self.__sub__(value)

    def __neg__(self):
        return subtract(0, self)

    def __mul__(self, value):
        return multiply(self, value)

    def __rmul__(self, value):
        return self.__mul__(value)

    def __imul__(self, value):
        return self.__mul__(value)

    def __truediv__(self, value):
        return divide(self, value)

    def __rtruediv__(self, value):
        return divide(value, self)

    def __itruediv__(self, value):
        return divide(self, value)

    def __floordiv__(self, value):
        return floor_divide(self, value)

    def __matmul__(self, value):
        return matmul(self, value)

    def __rmatmul__(self, value):
        return self.__matmul__(value)

    def __imatmul__(self, value):
        return matmul(self, value)

    def __pow__(self, value):
        return power(self, value)

    def __ipow__(self, value):
       return self.__pow__(value)

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

    def backward(self, grad):
        if self._op is not None:
            self._op.backward(grad)

    def reshape(self, *args, **kwargs):
        self.array = self.array.reshape(*args, **kwargs)
        if self.grad is not None:
            self.grad.array = self.grad.array.reshape(*args, **kwargs)
        return self

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.array))
        self._op.zero_grad() if self._op is not None else None

    def to_numpy(self):
        return self.array

    def copy(self):
        return deepcopy(self)

    def is_scalar(self):
        return self.array.size == 1

def tensor(object, requires_grad=False, *args, **kwargs):
    return Tensor(object, requires_grad, *args, **kwargs)

def zeros(*args, **kwargs):
    return Tensor(np.zeros(*args, **kwargs))

def ones(*args, **kwargs):
    return Tensor(np.ones(*args, **kwargs))

def zeros_like(a, *args, **kwargs):
    return Tensor(np.zeros_like(a.array if isinstance(a, Tensor) else np.array(a), *args, **kwargs))

def ones_like(a, *args, **kwargs):
    return Tensor(np.ones_like(a.array if isinstance(a, Tensor) else np.array(a), *args, **kwargs))

class Parameter(Tensor):
    def __init__(self, object, *args, **kwargs):
        super().__init__(object, *args, **kwargs)
        self.momentums = zeros_like(self)
        self.cache = zeros_like(self)
        self.requires_grad = True

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.array))
        self.momentums.zero_grad()
        self.cache.zero_grad()
        self._op.zero_grad() if self._op is not None else None

def parameter(object, *args, **kwargs):
    return Parameter(object, *args, **kwargs)

class Operation:
    def __init__(self, x1, *args, **kwargs):
        self.x1 = x1
        self.output = None
        self.args = args
        self.kwargs = kwargs
        self.__dict__.update(kwargs)

    def __call__(self):
        pass

    def zero_grad(self):
        if isinstance(self.x1, Tensor) and self.x1.requires_grad:
            self.x1.zero_grad()
        if hasattr(self, "x2") and isinstance(self.x2, Tensor) and self.x2.requires_grad:
            self.x2.zero_grad()

    @staticmethod
    def _get_array(obj):
        if isinstance(obj, Tensor):
            return obj.array
        else:
            return np.array(obj)

    @staticmethod
    def _apply_grad(obj, grad):
        if isinstance(obj, Tensor) and obj.requires_grad:
            if obj.grad is None: obj.grad = zeros_like(obj)
            grad = grad if isinstance(grad, Tensor) else Tensor(grad)
            obj.grad += grad  
            obj.backward(grad)

    @staticmethod
    def _requires_grad(*inputs):
        return any(isinstance(i, Tensor) and i.requires_grad for i in inputs)

    def _compress_grad(self, grad, tensor):
        grad = np.atleast_2d(grad.array if isinstance(grad, Tensor) else grad)
        array = self._get_array(tensor)
        extra_dims = np.array(grad).ndim - array.ndim
        for _ in range(extra_dims):
            grad = np.sum(grad, axis=0)
        for i, dim in enumerate(array.shape):
            if dim == 1:
                grad = np.sum(grad, axis=i, keepdims=True)
        return Tensor(grad)  # Return as Tensor

class _Sum(Operation):
    def __init__(self, x1, axis=None, keepdims=False, *args, **kwargs):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x1, *args,  **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(
            np.sum(
                x1_array,
                axis=self.axis,
                keepdims=self.keepdims,
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1)
        )
        return self.output

    def backward(self, grad):
        grad_array = self._get_array(grad)
        if self.axis is not None and self.keepdims == False:
            grad_array = np.broadcast_to(grad.array if isinstance(grad, Tensor) else grad, self.x1.shape)
            self._apply_grad(self.x1, Tensor(grad_array))
        else:
            self._apply_grad(self.x1, self._compress_grad(grad_array, self.x1))

def sum(x1, axis=None, keepdims=False, *args, **kwargs):
    return _Sum(x1, axis=axis, keepdims=keepdims, *args, **kwargs)()

class _Add(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, *args,  **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(
            np.add(
                x1_array,
                x2_array,
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1, self.x2)
        )
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        grad_array = self._get_array(grad)
        self._apply_grad(self.x1, self._compress_grad(grad_array, x1_array))
        self._apply_grad(self.x2, self._compress_grad(grad_array, x2_array))

def add(x1, x2, *args, **kwargs):
    return _Add(x1, x2, *args, **kwargs)()

class _Subtract(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, *args,  **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(
            np.subtract(
                x1_array,
                x2_array,
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        grad_array = self._get_array(grad)
        self._apply_grad(self.x1, self._compress_grad(grad_array, self.x1))
        self._apply_grad(self.x2, -self._compress_grad(grad_array, self.x2))

def subtract(x1, x2, *args, **kwargs):
    return _Subtract(x1, x2, *args, **kwargs)()

class _Multiply(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, *args,  **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(
            np.multiply(
                x1_array,
                x2_array,
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        grad_array = self._get_array(grad)
        self._apply_grad(self.x1,  self._compress_grad(grad_array * x2_array, self.x1))
        self._apply_grad(self.x2,  self._compress_grad(grad_array * x1_array, self.x2))

def multiply(x1, x2, *args, **kwargs):
    return _Multiply(x1, x2, *args, **kwargs)()

class _Dot(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, *args, **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(
            np.dot(
                x1_array,
                x2_array,
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        grad_array = self._get_array(grad)
        self._apply_grad(self.x1, Tensor(np.dot(grad_array, x2_array.T)))
        self._apply_grad(self.x2, Tensor(np.dot(x1_array.T, grad_array)))

def dot(x1, x2, *args, **kwargs):
    return _Dot(x1, x2, *args, **kwargs)()

class _Matmul(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, *args,  **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(
            np.matmul(
                x1_array,
                x2_array,
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        grad_array = self._get_array(grad)
        self._apply_grad(self.x1, Tensor(np.matmul(grad_array, x2_array.T)))
        self._apply_grad(self.x2, Tensor(np.matmul(x1_array.T, grad_array)))

def matmul(x1, x2, *args, **kwargs):
    return _Matmul(x1, x2, *args, **kwargs)()

class _Divide(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1,  *args,  **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(
            np.divide(
                x1_array,
                x2_array,
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        grad_array = self._get_array(grad)
        self._apply_grad(self.x1, self._compress_grad(grad_array / x2_array, self.x1))
        self._apply_grad(self.x2, self._compress_grad(-grad_array * x1_array / (x2_array * x2_array), self.x2))

def divide(x1, x2, *args, **kwargs):
    return _Divide(x1, x2, *args, **kwargs)()

class _FloorDivide(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, *args,  **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(
            np.floor_divide(
                x1_array,
                x2_array,
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        grad_array = self._get_array(grad)
        self._apply_grad(self.x1, self._compress_grad(grad_array, self.x1))
        self._apply_grad(self.x2, self._compress_grad(grad_array, self.x2))

def floor_divide(x1, x2, *args, **kwargs):
    return _FloorDivide(x1, x2, *args, **kwargs)()

class _Power(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1, *args,  **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(
            np.power(
                np.clip(x1_array, _F64_TINY, _F64_MAX),
                np.clip(x2_array, None, _F64_MAX_EXP_ARG),
                *self.args, **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1, self.x2)
        )
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        grad_array = grad.array if isinstance(grad, Tensor) else grad
        self._apply_grad(self.x1, Tensor(grad_array *
                         x2_array *
                         np.power(
                             np.clip(x1_array, _F64_TINY, _F64_MAX),
                             np.clip(x2_array, None, _F64_MAX_EXP_ARG) - 1
                         ))
        )
        self._apply_grad(self.x2, Tensor(grad_array *
                         np.power(
                             np.clip(x1_array, _F64_TINY, _F64_MAX),
                             np.clip(x2_array, None, _F64_MAX_EXP_ARG)
                         ) *
                        np.log(np.clip(x1_array, _F64_TINY, None)))
        )

def power(x1, x2, *args, **kwargs):
    return _Power(x1, x2, *args, **kwargs)()

class _Exp(Operation):
    def __init__(self, x1, *args, **kwargs):
        super().__init__(x1, *args,  **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(
            np.exp(
                np.clip(x1_array, None, _F64_MAX_EXP_ARG),
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        grad_array = grad.array if isinstance(grad, Tensor) else grad
        self._apply_grad(self.x1, Tensor(grad_array * np.exp(np.clip(x1_array, None, _F64_MAX_EXP_ARG))))

def exp(x1, *args, **kwargs):
    return _Exp(x1, *args, **kwargs)()

class _Log(Operation):
    def __init__(self, x1, *args, **kwargs):
        super().__init__(x1, *args,  **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(
            np.log(
                np.clip(x1_array, _F64_TINY, None),
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        grad_array = grad.array if isinstance(grad, Tensor) else grad
        self._apply_grad(self.x1, Tensor(grad_array / np.clip(x1_array, _F64_TINY, None)))

def log(x1, *args, **kwargs):
    return _Log(x1, *args, **kwargs)()

class _Max(Operation):
    def __init__(self, x1, axis=None, keepdims=False, *args, **kwargs):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x1, *args,  **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(
            np.max(
                x1_array,
                axis=self.axis,
                keepdims=self.keepdims,
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        grad_array = grad.array if isinstance(grad, Tensor) else grad
        mask = (x1_array.max(axis=self.axis, keepdims=True) == x1_array).astype(int)
        self._apply_grad(self.x1, Tensor(grad_array * mask))

def max(x1, axis=None, keepdims=False, *args, **kwargs):
    return _Max(x1, axis=axis, keepdims=keepdims, *args, **kwargs)()

class _Min(Operation):
    def __init__(self, x1, axis=None, keepdims=False, *args, **kwargs):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x1, *args,  **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(
            np.min(
                x1_array,
                axis=self.axis,
                keepdims=self.keepdims,
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        grad_array = grad.array if isinstance(grad, Tensor) else grad
        mask = (x1_array.min(axis=self.axis, keepdims=True) == x1_array).astype(int)
        self._apply_grad(self.x1, Tensor(grad_array * mask))

def min(x1, axis=None, keepdims=False, *args, **kwargs):
    return _Min(x1, axis=axis, keepdims=keepdims, *args, **kwargs)()

class _Maximum(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1,  *args, **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(
            np.maximum(
                x1_array,
                x2_array,
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        output_array = self._get_array(self.output)
        grad_array = grad.array if isinstance(grad, Tensor) else grad
        mask1 = (output_array == x1_array).astype(int)
        mask2 = (output_array == x2_array).astype(int)
        self._apply_grad(self.x1, Tensor(grad_array * mask1))
        self._apply_grad(self.x2, Tensor(grad_array * mask2))

def maximum(x1, x2, *args, **kwargs):
    return _Maximum(x1, x2, *args, **kwargs)()

class _Minimum(Operation):
    def __init__(self, x1, x2, *args, **kwargs):
        self.x2 = x2
        super().__init__(x1,  *args, **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        self.output = Tensor(
            np.minimum(
                x1_array,
                x2_array,
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1, self.x2))
        return self.output

    def backward(self, grad):
        x1_array = self._get_array(self.x1)
        x2_array = self._get_array(self.x2)
        output_array = self._get_array(self.output)
        grad_array = grad.array if isinstance(grad, Tensor) else grad
        mask1 = (output_array == x1_array).astype(int)
        mask2 = (output_array == x2_array).astype(int)
        self._apply_grad(self.x1, Tensor(grad_array * mask1))
        self._apply_grad(self.x2, Tensor(grad_array * mask2))

def minimum(x1, x2, *args, **kwargs):
    return _Minimum(x1, x2, *args, **kwargs)()

class _Argmax(Operation):
    def __init__(self, x1, *args, **kwargs):
        super().__init__(x1,  *args,  **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(
            np.argmax(
                x1_array,
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        grad_array = self._get_array(grad)
        self._apply_grad(self.x1, grad_array)

def argmax(x1, *args, **kwargs):
    return _Argmax(x1,  *args, **kwargs)()

class _Argmin(Operation):
    def __init__(self, x1, x2=None, *args, **kwargs):
        super().__init__(x1, x2, *args,  **kwargs)

    def __call__(self, *args, **kwargs):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(
            np.argmin(
                x1_array,
                *args,
                **kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        grad_array = self._get_array(grad)
        self._apply_grad(self.x1, grad_array)

def argmin(x1, *args, **kwargs):
    return _Argmin(x1, *args, **kwargs)()

class _Clip(Operation):
    def __init__(self, x1, *args, **kwargs):
        super().__init__(x1, *args,  **kwargs)

    def __call__(self):
        x1_array = self._get_array(self.x1)
        self.output = Tensor(
            np.clip(
                x1_array,
                *self.args,
                **self.kwargs
            ), _operation=self, requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        grad_array = self._get_array(grad)
        self._apply_grad(self.x1, grad_array)

def clip(x1, *args, **kwargs):
    return _Clip(x1,  *args, **kwargs)()

class Function(Operation):
    def __init__(self, x1):
        super().__init__(x1)

    def __call__(self):
        pass

    def update_requires_grad(self, obj, value):
        if not isinstance(value, bool):
            raise TypeError('`value` must be bool.')
        if isinstance(obj, Tensor):
            obj.requires_grad = value
            self.__call__()
        else:
            raise TypeError('`obj` must be Tensor.')

    def backward(self, grad):
        self.output.backward(grad)

    def derivative(self): # Must be used independently of a computational graph.
        # TODO Needs improvements
        x1 = self.x1
        self.x1 = Tensor(self.x1.array)
        self.update_requires_grad(self.x1, True)
        self.backward(1)
        grad = self.x1.grad
        self.x1 = x1
        return grad

class _Sigmoid(Function):
    def __init__(self, x1):
        super().__init__(x1)

    def __call__(self):
        self.output = 1 / (1 + exp(-self.x1))
        return self.output

def sigmoid(x, derivative=False):
    return _Sigmoid(x)() if not derivative else _Sigmoid(x).derivative()

class _Tanh(Function):
    def __init__(self, x1):
        super().__init__(x1)

    def __call__(self):
        self.output = (exp(self.x1) - exp(-self.x1)) / (exp(self.x1) + exp(-self.x1))
        return self.output

def tanh(x, derivative=False):
    return _Tanh(x)() if not derivative else _Tanh(x).derivative()

class _Softmax(Function):
    def __init__(self, x1):
        super().__init__(x1)

    def __call__(self):
        exp_values = exp(self.x1 - max(self.x1, axis=1, keepdims=True))
        probabilities = exp_values / sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

def softmax(x, derivative=False):
    return _Softmax(x)() if not derivative else _Softmax(x).derivative()

class _ReLU(Function):
    def __init__(self, x1):
        super().__init__(x1)

    def __call__(self):
        self.output = maximum(0, self.x1)
        return self.output

def relu(x, derivative=False):
    return _ReLU(x)() if not derivative else _ReLU(x).derivative()

class _LeakyReLU(Function):
    def __init__(self, x1):
        super().__init__(x1)

    def __call__(self):
        self.output = Tensor(
            np.where(
                self.x1.array > 0,
                self.x1.array,
                self.x1.array * 0.01
            ), _operation=self, requires_grad=self._requires_grad(self.x1))
        return self.output

    def backward(self, grad):
        grad_array = grad.array if isinstance(grad, Tensor) else grad
        self._apply_grad(self.x1, Tensor(grad_array * self.derivative()))

    def derivative(self):
        self.update_requires_grad(self.x1, True)
        dA = np.ones_like(self.x1.array)
        dA[self.x1.array < 0] = 0.01
        self.update_requires_grad(self.x1, False)
        return Tensor(dA)

def leaky_relu(x, derivative=False):
    return _LeakyReLU(x)() if not derivative else _LeakyReLU(x).derivative()