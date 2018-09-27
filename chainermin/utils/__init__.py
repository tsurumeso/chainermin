import numpy


def force_array(x):
    if numpy.isscalar(x):
        return numpy.array(x)
    else:
        return x
