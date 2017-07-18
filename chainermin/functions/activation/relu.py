import numpy

from chainermin import function


class ReLU(function.Function):

    def forward(self, x):
        return numpy.maximum(x[0], 0),

    def backward(self, x, gy):
        return gy[0] * (x[0] > 0),


def relu(x):
    return ReLU()(x)
