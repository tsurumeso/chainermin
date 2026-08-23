from chainermin import function
from chainermin import backend


class ReLU(function.Function):

    def forward(self, x):
        xp = backend.get_array_module(*x)
        return xp.maximum(x[0], 0),

    def backward(self, x, gy):
        return gy[0] * (x[0] > 0),


def relu(x):
    return ReLU()(x)
