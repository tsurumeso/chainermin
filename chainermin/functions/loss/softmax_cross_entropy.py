import numpy

from chainermin import function
from chainermin import backend


def _log_softmax(x):
    m = x.max(axis=1, keepdims=True)
    y = x - m
    z = numpy.log(numpy.exp(y).sum(axis=1, keepdims=True))
    return y - z


class SoftmaxCrossEntropy(function.FunctionNode):

    def forward(self, inputs):
        self.retain_inputs()
        xp = backend.get_array_module(*inputs)
        x, t = inputs
        n_batches = x.shape[0]
        n_classes = x.shape[1]
        self.t = xp.array([t == i for i in range(n_classes)]).astype(numpy.int32).T
        self.log_p = _log_softmax(x)
        y = -xp.sum(self.t * self.log_p)
        y /= n_batches
        return y.reshape(()),

    def backward(self, inputs, grad_outputs):
        xp = backend.get_array_module(*inputs, *grad_outputs)
        x, t = inputs
        n_batches = x.shape[0]
        gx = xp.exp(self.log_p) - self.t
        gx /= n_batches
        return gx, None


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)
    