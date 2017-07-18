import numpy


class Initializer(object):

    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, shape):
        raise NotImplementedError()


class Normal(Initializer):

    def __init__(self, scale=1.0, dtype=numpy.float32):
        self.scale = scale
        super(Normal, self).__init__(dtype)

    def __call__(self, shape):
        array = numpy.empty(shape, dtype=self.dtype)
        array[...] = numpy.random.normal(0.0, self.scale, size=shape)
        return array


class Constant(Initializer):

    def __init__(self, fill_value=0, dtype=numpy.float32):
        self.fill_value = fill_value
        super(Constant, self).__init__(dtype)

    def __call__(self, shape):
        array = numpy.empty(shape, dtype=self.dtype)
        array[...] = numpy.asarray(self.fill_value)
        return array
