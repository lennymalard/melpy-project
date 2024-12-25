import numpy as np

class Tensor:
    def __init__(self, object, dtype=None, *,copy=True, order="K", subok=False, ndim=0, like=None):
        self.array = np.array(object, dtype=dtype, copy=copy, order=order, subok=subok)
        self.requires_grad = False
        self.grad = None

    def __str__(self):
        return self.array.__str__()

    def __add__(self, value):
        return Tensor(self.array + value.array)

    def __iadd__(self, value):
        self.array += value.array
        return self

    def __sub__(self, value):
        return Tensor(self.array - value.array)

    def __isub__(self, value):
        self.array -= value.array.array
        return self

    def __mul__(self, value):
        return Tensor(self.array * value.array)

    def __imul__(self, value):
        self.array *= value.array
        return self

    def __truediv__(self, value):
        return Tensor(self.array / value.array)

    def __itruediv__(self, value):
        self.array = self.array.astype(np.float64) / value.array.astype(np.float64)
        return self

    def __matmul__(self, value):
        return self.array @ value.array

    def __imatmul__(self, value):
        self.array @= value.array
        return self

