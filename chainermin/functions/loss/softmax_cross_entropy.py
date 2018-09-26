import numpy

from chainermin import function


def _log_softmax(x):
    m = x.max(axis=1, keepdims=True)
    y = x - m
    z = numpy.log(numpy.exp(y).sum(axis=1, keepdims=True))
    return y - z


class SoftmaxCrossEntropy(function.Function):

    def forward(self, inputs):
        x, t = inputs
        self.t = numpy.array([t == i for i in range(10)]).astype(numpy.int32).T
        self.log_p = _log_softmax(x)
        y = -numpy.sum(self.t * self.log_p)
        return y.reshape(()),

    def backward(self, inputs, grad_outputs):
        x, t = inputs
        gx = numpy.exp(self.log_p) - self.t
        return gx, None


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)
