import numpy

from chainermin import function
from chainermin.utils import conv


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class Convolution2DFunction(function.Function):

    def __init__(self, stride=1, pad=0, cover_all=False):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.cover_all = cover_all

    def forward(self, inputs):
        x, W, b = inputs[:3]

        kh, kw = W.shape[2:]
        self.col = conv.im2col_cpu(
            x, kh, kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all)
        y = numpy.tensordot(
            self.col, W, ((1, 2, 3), (1, 2, 3))).astype(x.dtype, copy=False)
        y += b
        return numpy.rollaxis(y, 3, 1),

    def backward(self, inputs, grad_outputs):
        x, W = inputs[:2]

        gy = grad_outputs[0]
        h, w = x.shape[2:]

        gW = numpy.tensordot(
            gy, self.col, ((0, 2, 3), (0, 4, 5))).astype(W.dtype, copy=False)

        gcol = numpy.tensordot(W, gy, (0, 1)).astype(x.dtype, copy=False)
        gcol = numpy.rollaxis(gcol, 3)
        gx = conv.col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)

        gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def convolution_2d(x, W, b=None, stride=1, pad=0, cover_all=False):
    func = Convolution2DFunction(stride, pad, cover_all)
    return func(x, W, b)
