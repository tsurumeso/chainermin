import numpy

from chainermin import initializer


class Normal(initializer.Initializer):

    def __init__(self, scale=1.0, dtype=numpy.float32):
        self.scale = scale
        super(Normal, self).__init__(dtype)

    def __call__(self, shape):
        array = numpy.empty(shape, dtype=self.dtype)
        array[...] = numpy.random.normal(0.0, self.scale, size=shape)
        return array
