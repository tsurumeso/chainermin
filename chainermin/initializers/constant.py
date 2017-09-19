import numpy

from chainermin import initializer


class Constant(initializer.Initializer):

    def __init__(self, fill_value=0, dtype=numpy.float32):
        self.fill_value = fill_value
        super(Constant, self).__init__(dtype)

    def __call__(self, array):
        array[...] = numpy.asarray(self.fill_value)
        return array
