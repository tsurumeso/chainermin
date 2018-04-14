import numpy

from chainermin import function


class Softmax(function.Function):

    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = x[0] - x[0].max(axis=self.axis, keepdims=True)
        numpy.exp(y, out=y)
        y /= y.sum(axis=self.axis, keepdims=True)
        self._x_shape = x[0].shape
        return y,

    def backward(self, x, gy):
        y = self.output_data[0]
        gx = y * gy[0]
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx,


def softmax(x, axis=1):
    return Softmax(axis=axis)(x)
