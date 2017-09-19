import numpy


class Initializer(object):

    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, shape):
        raise NotImplementedError()


def get_fans(shape):
    if not isinstance(shape, tuple):
        raise ValueError('shape must be tuple')

    if len(shape) < 2:
        raise ValueError('shape must be of length >= 2: shape={}', shape)

    fan_in = numpy.prod(shape[1:])
    fan_out = shape[0]
    return fan_in, fan_out
