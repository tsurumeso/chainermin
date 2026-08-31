import numpy

from chainermin.initializers.constant import Constant  # NOQA
from chainermin.initializers.normal import (
    HeNormal,  # NOQA
    Normal,  # NOQA
)


def generate_array(initializer, shape):
    array = numpy.empty(shape, dtype=numpy.float32)
    initializer(array)
    return array
