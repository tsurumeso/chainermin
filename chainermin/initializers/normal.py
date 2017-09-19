import numpy

from chainermin import initializer


class Normal(initializer.Initializer):

    def __init__(self, scale=1.0, dtype=numpy.float32):
        self.scale = scale
        super(Normal, self).__init__(dtype)

    def __call__(self, array):
        array[...] = numpy.random.normal(0.0, self.scale, size=array.shape)
        return array


class HeNormal(initializer.Initializer):

    def __init__(self, scale=1.0, dtype=None):
        self.scale = scale
        super(HeNormal, self).__init__(dtype)

    def __call__(self, array):
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(2. / fan_in)
        Normal(s)(array)
