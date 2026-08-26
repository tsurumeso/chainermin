import numpy

from chainermin import function


class Transpose(function.FunctionNode):

    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, inputs):
        x = inputs[0]
        return x.transpose(self.axes),

    def backward(self, inputs, gy):
        inv_axes = self.axes
        if inv_axes:
            axes_len = len(inv_axes)
            inv_axes = tuple(numpy.argsort([ax % axes_len for ax in inv_axes]))
        gx = gy[0].transpose(inv_axes)
        return gx,


def transpose(x, axes=None):
    """Permute the dimensions of an input variable without copy.
    """
    return Transpose(axes)(x)
