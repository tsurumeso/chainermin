import math

from chainermin.functions.connection import convolution_2d
from chainermin import initializers
from chainermin import link


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class Convolution2D(link.Link):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1):
        super(Convolution2D, self).__init__()
        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.out_channels = out_channels

        kh, kw = _pair(self.ksize)
        W_shape = (self.out_channels, in_channels, kh, kw)

        W_initializer = initializers.HeNormal(math.sqrt(wscale))
        bias_initializer = initializers.Constant(0)
        self.add_param('W', W_shape, initializer=W_initializer)
        self.add_param('b', out_channels, initializer=bias_initializer)

    def __call__(self, x):
        return convolution_2d.convolution_2d(
            x, self.W, self.b, self.stride, self.pad)
