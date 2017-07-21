import numpy

from chainermin import function
from chainermin.functions.activation import log_softmax


class SoftmaxCrossEntropy(function.Function):

    def __init__(self, normalize=True, cache_score=True, ignore_label=-1):
        self.normalize = normalize
        self.cache_score = cache_score
        self.ignore_label = ignore_label

    def forward(self, inputs):
        x, t = inputs
        log_y = log_softmax._log_softmax(x)
        if self.cache_score:
            self.y = numpy.exp(log_y)
        log_yd = numpy.rollaxis(log_y, 1)
        log_yd = log_yd.reshape(len(log_yd), -1)
        log_p = log_yd[numpy.maximum(t.ravel(), 0), numpy.arange(t.size)]

        log_p *= (t.ravel() != self.ignore_label)
        if self.normalize:
            count = (t != self.ignore_label).sum()
        else:
            count = len(x)
        self._coeff = 1.0 / max(count, 1)

        y = log_p.sum(keepdims=True) * (-self._coeff)
        return y.reshape(()),

    def backward(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        if hasattr(self, 'y'):
            y = self.y.copy()
        else:
            y = log_softmax._log_softmax(x)
            numpy.exp(y, out=y)

        if y.ndim == 2:
            gx = y
            gx[numpy.arange(len(t)), numpy.maximum(t, 0)] -= 1
            gx *= (t != self.ignore_label).reshape((len(t), 1))
        else:
            n_unit = t.size // len(t)
            gx = y.reshape(y.shape[0], y.shape[1], -1)
            fst_index = numpy.arange(t.size) // n_unit
            trd_index = numpy.arange(t.size) % n_unit
            gx[fst_index, numpy.maximum(t.ravel(), 0), trd_index] -= 1
            gx *= (t != self.ignore_label).reshape((len(t), 1, -1))
            gx = gx.reshape(y.shape)

        gx *= gloss * self._coeff
        return gx, None


def softmax_cross_entropy(
        x, t, normalize=True, cache_score=True,
        ignore_label=-1):
    return SoftmaxCrossEntropy(normalize, cache_score, ignore_label)(x, t)
