import numpy

from chainer import function


def _log_softmax(x):
    log_z = x.max(axis=1, keepdims=True)
    y = x - log_z
    numpy.exp(y, out=y)
    s = y.sum(axis=1, keepdims=True)
    numpy.log(s, out=s)
    log_z += s
    y = x - log_z
    return y


class LogSoftmax(function.Function):

    def forward(self, xs):
        y = _log_softmax(xs[0])
        self._x_shape = xs[0].shape
        self._x_dtype = xs[0].dtype
        return y,

    def backward(self, x, gy):
        y = self.output_data[0]
        gx = gy[0] - numpy.exp(y) * gy[0].sum(axis=1, keepdims=True)
        return gx,


def log_softmax(x):
    return LogSoftmax()(x)
