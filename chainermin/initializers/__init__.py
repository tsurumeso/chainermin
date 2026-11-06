import numpy

from chainermin.initializers.constant import Constant  # NOQA
from chainermin.initializers.normal import (
    HeNormal,  # NOQA
    Normal,  # NOQA
)


def generate_array(initializer, shape, dtype=numpy.float32):
    if getattr(initializer, "dtype", None) is not None:
        dtype = initializer.dtype
    array = numpy.empty(shape, dtype=dtype)
    initializer(array)
    return array
