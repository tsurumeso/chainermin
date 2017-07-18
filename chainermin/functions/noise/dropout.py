import numpy

from chainermin import function


class Dropout(function.Function):

    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def forward(self, inputs):
        x = inputs[0]
        if not hasattr(self, 'mask'):
            scale = x.dtype.type(1. / (1 - self.dropout_ratio))
            flag = numpy.random.rand(*x.shape) >= self.dropout_ratio
            self.mask = scale * flag
        return x * self.mask,

    def backward(self, inputs, grad_outputs):
        gy = grad_outputs[0]
        return gy * self.mask,


def dropout(x, ratio=0.5, train=True):
    if train:
        return Dropout(ratio)(x)
    return x
