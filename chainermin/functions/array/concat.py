import numpy

from chainermin import function


class Concat(function.Function):

    # concat along the channel dimension by default
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, xs):
        return numpy.concatenate(xs, axis=self.axis),

    def backward(self, xs, gy):
        sizes = numpy.array([x.shape[self.axis] for x in xs[:-1]]).cumsum()
        return numpy.split(gy[0], sizes, axis=self.axis)


def concat(xs, axis=1):
    """Concatenates given variables along an axis.
    """
    return Concat(axis=axis)(*xs)
