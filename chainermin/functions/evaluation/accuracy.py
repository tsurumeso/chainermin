import numpy

from chainermin import function


class Accuracy(function.Function):
    
    def forward(self, inputs):
        y, t = inputs
        pred = y.argmax(axis=1).reshape(t.shape)
        return numpy.asarray((pred == t).mean(dtype=y.dtype)),


def accuracy(y, t):
    return Accuracy()(y, t)