import numpy

from chainermin import function


class Sigmoid(function.Function):

    def forward(self, x):
        self.y = numpy.tanh(x[0] * 0.5) * 0.5 + 0.5
        return self.y,

    def backward(self, x, gy):
        return gy[0] * self.y * (1 - self.y),


def sigmoid(x):
    return Sigmoid()(x)
